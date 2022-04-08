import torch
import torch.nn as nn
from ding.torch_utils import MLP, get_lstm, Transformer
from ding.utils import list_split
from ding.model.common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


# TODO: 选择['compute_actor', 'compute_critic', 'compute_actor_critic']三个函数的方法非常巧妙


class RelationGCN(nn.Module):

    def __init__(
            self,
            hidden_shape: int,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(RelationGCN, self).__init__()
        # activation
        self.act = activation
        # layers
        self.thorn_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.clone_relation_layers = MLP(
            hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.agg_relation_layers = MLP(
            4 * hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )

    def forward(self, food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask):
        b, t, c = clone.shape[0], thorn_relation.shape[2], clone.shape[1]
        # encode thorn relation
        thorn_relation = self.thorn_relation_layers(thorn_relation) * thorn_mask.view(b, 1, t, 1)  # [b,n_clone,n_thorn,c]
        thorn_relation = thorn_relation.max(2).values # [b,n_clone,c]
        # encode clone relation
        clone_relation = self.clone_relation_layers(clone_relation) * clone_mask.view(b, 1, c, 1) # [b,n_clone,n_clone,c]
        clone_relation = clone_relation.max(2).values # [b,n_clone,c]
        # encode aggregated relation
        agg_relation = torch.cat([clone, food_relation, thorn_relation, clone_relation], dim=2)
        clone = self.agg_relation_layers(agg_relation)
        return clone



class EncoderCritic(nn.Module):
    def __init__(
            self,
            scalar_shape: int,
            food_shape: int,
            food_relation_shape: int,
            thorn_relation_shape: int,
            clone_shape: int,
            clone_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            activation=nn.ReLU(inplace=True)
    ) -> None:
        super(EncoderCritic, self).__init__()

        # scalar encoder
        self.scalar_encoder = MLP(
            scalar_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # food encoder
        layers = []
        kernel_size = [5, 3, 1]
        stride = [4, 2, 1]
        shape = [hidden_shape // 4, hidden_shape // 2, hidden_shape]
        input_shape = food_shape
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_shape, shape[i], kernel_size[i], stride[i], kernel_size[i] // 2))
            layers.append(activation)
            input_shape = shape[i]
        self.food_encoder = nn.Sequential(*layers)
        # food relation encoder
        self.food_relation_encoder = MLP(
            food_relation_shape, hidden_shape // 2, hidden_shape, layer_num=2, activation=activation
        )
        # thorn relation encoder
        self.thorn_relation_encoder = MLP(
            thorn_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone encoder
        self.clone_encoder = MLP(
            clone_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone relation encoder
        self.clone_relation_encoder = MLP(
            clone_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # gcn
        self.gcn = RelationGCN(
            hidden_shape, activation=activation
        )
        self.agg_encoder = MLP(
            3 * hidden_shape, hidden_shape, encode_shape, layer_num=2, activation=activation
        )

    def forward(self, scalar, food, food_relation, thorn_relation, thorn_mask, clone, clone_relation, clone_mask):
        # encode scalar
        scalar = self.scalar_encoder(scalar)  # [b,c]
        # encode food
        food = self.food_encoder(food)  # [b,c,h,w]
        food = food.reshape(*food.shape[:2], -1).max(-1).values  # [b,c]
        # encode food relation
        food_relation = self.food_relation_encoder(food_relation)  # [b,c]
        # encode thorn relation
        thorn_relation = self.thorn_relation_encoder(thorn_relation)  # [b,n_clone,n_thorn, c]
        # encode clone
        clone = self.clone_encoder(clone)  # [b,n_clone,c]
        # encode clone relation
        clone_relation = self.clone_relation_encoder(clone_relation)  # [b,n_clone,n_clone,c]
        # aggregate all relation
        clone = self.gcn(food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask)
        clone = clone.max(1).values  # [b,c]
        encoder = self.agg_encoder(torch.cat([scalar, food, clone], dim=1))
        # x1 = torch.tanh(self.linear_q_1(encoder))
        # q_eval = torch.tanh(self.linear_q_2(x1))
        return encoder


class EncoderActor(nn.Module):
    def __init__(
            self,
            scalar_shape: int,
            food_shape: int,
            food_relation_shape: int,
            thorn_relation_shape: int,
            clone_shape: int,
            clone_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            activation=nn.ReLU(inplace=True),
    ) -> None:
        super(EncoderActor, self).__init__()

        # scalar encoder
        self.scalar_encoder = MLP(
            scalar_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # food encoder
        layers = []
        kernel_size = [5, 3, 1]
        stride = [4, 2, 1]
        shape = [hidden_shape // 4, hidden_shape // 2, hidden_shape]
        input_shape = food_shape
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_shape, shape[i], kernel_size[i], stride[i], kernel_size[i] // 2))
            layers.append(activation)
            input_shape = shape[i]
        self.food_encoder = nn.Sequential(*layers)
        # food relation encoder
        self.food_relation_encoder = MLP(
            food_relation_shape, hidden_shape // 2, hidden_shape, layer_num=2, activation=activation
        )
        # thorn relation encoder
        self.thorn_relation_encoder = MLP(
            thorn_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone encoder
        self.clone_encoder = MLP(
            clone_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone relation encoder
        self.clone_relation_encoder = MLP(
            clone_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # gcn
        self.gcn = RelationGCN(
            hidden_shape, activation=activation
        )
        self.agg_encoder = MLP(
            3 * hidden_shape, hidden_shape, encode_shape, layer_num=2, activation=activation
        )

    def forward(self, scalar, food, food_relation, thorn_relation, thorn_mask, clone, clone_relation, clone_mask):
        # encode scalar
        scalar = self.scalar_encoder(scalar)  # [b,c]
        # encode food
        food = self.food_encoder(food)  # [b,c,h,w]
        food = food.reshape(*food.shape[:2], -1).max(-1).values  # [b,c]
        # encode food relation
        food_relation = self.food_relation_encoder(food_relation)  # [b,c]
        # encode thorn relation
        thorn_relation = self.thorn_relation_encoder(thorn_relation)  # [b,n_clone,n_thorn, c]
        # encode clone
        clone = self.clone_encoder(clone)  # [b,n_clone,c]
        # encode clone relation
        clone_relation = self.clone_relation_encoder(clone_relation)  # [b,n_clone,n_clone,c]
        # aggregate all relation
        clone = self.gcn(food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask)
        clone = clone.max(1).values  # [b,c]

        return self.agg_encoder(torch.cat([scalar, food, clone], dim=1))


class GoBiggerLinear(nn.Module):
    def __init__(self, encode_shape):
        super(GoBiggerLinear, self).__init__()
        self.linear_q_1 = nn.Linear(encode_shape, 256)
        self.linear_q_2 = nn.Linear(256, 1)

    def forward(self, encoder: torch.Tensor):
        # x = torch.tanh(self.linear_q_1(encoder))
        # q_va = torch.tanh(self.linear_q_2(x))
        x = self.linear_q_1(encoder)
        q_va = self.linear_q_2(x)
        return {'pred': q_va}


class GoBiggerPPoModel(nn.Module):

    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(self,
                 scalar_shape: int,
                 food_shape: int,
                 food_relation_shape: int,
                 thorn_relation_shape: int,
                 clone_shape: int,
                 clone_relation_shape: int,
                 hidden_shape: int,
                 encode_shape: int,
                 action_type_shape: int,
                 rnn: bool = False,
                 activation=nn.ReLU(inplace=True),
            ) -> None:
        super(GoBiggerPPoModel, self).__init__()
        self.activation = activation
        self.action_type_shape = action_type_shape
        self.actor_encoder = EncoderActor(scalar_shape = scalar_shape,
                                           food_shape = food_shape,
                                           food_relation_shape = food_relation_shape,
                                           thorn_relation_shape = thorn_relation_shape,
                                           clone_shape = clone_shape,
                                           clone_relation_shape = clone_relation_shape,
                                           hidden_shape = hidden_shape,
                                           encode_shape = encode_shape,
                                           activation = activation)

        self.critic_encoder = EncoderCritic(scalar_shape = scalar_shape,
                                           food_shape = food_shape,
                                           food_relation_shape = food_relation_shape,
                                           thorn_relation_shape = thorn_relation_shape,
                                           clone_shape = clone_shape,
                                           clone_relation_shape = clone_relation_shape,
                                           hidden_shape = hidden_shape,
                                           encode_shape = encode_shape,
                                           activation = activation)
        self.actor_head = DiscreteHead(32, action_type_shape, layer_num=2, activation=self.activation)
        self.critic_head = GoBiggerLinear(encode_shape = encode_shape)

        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]

        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

    def forward(self, inputs, mode:str):

        assert mode in self.mode, "not support forward mode:{}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor):
        B = inputs['batch']
        A = inputs['player_num_per_team']

        scalar = inputs['scalar']
        food = inputs['food']
        food_relation = inputs['food_relation']
        thorn_relation = inputs['thorn_relation']
        thorn_mask = inputs['thorn_mask']
        clone = inputs['clone']
        clone_relation = inputs['clone_relation']
        clone_mask = inputs['clone_mask']

        x = self.actor_encoder(scalar, food, food_relation, thorn_relation, thorn_mask, clone,
                                clone_relation, clone_mask)
        res = self.actor_head(x)

        action_type_logit = res['logit']  # B, M, action_type_size
        action_type_logit = action_type_logit.reshape(B, A, *action_type_logit.shape[1:])

        return {'logit': action_type_logit,}

    def compute_critic(self, inputs: torch.Tensor):
        B = inputs['batch']
        A = inputs['player_num_per_team']
        scalar = inputs['scalar']
        food = inputs['food']
        food_relation = inputs['food_relation']
        thorn_relation = inputs['thorn_relation']
        thorn_mask = inputs['thorn_mask']
        clone = inputs['clone']
        clone_relation = inputs['clone_relation']
        clone_mask = inputs['clone_mask']

        x = self.critic_encoder(scalar, food, food_relation, thorn_relation, thorn_mask, clone,
                                             clone_relation, clone_mask)
        value = self.critic_head(x)
        value_pred = value['pred']
        value_type_pred = value_pred.reshape(B, A, *value_pred.shape[1:])
        value_output_pred = torch.mean(value_type_pred, 1)

        return {'value': value_output_pred}

    def compute_actor_critic(self, inputs:torch.Tensor):
        B = inputs['batch']
        A = inputs['player_num_per_team']

        scalar = inputs['scalar']
        food = inputs['food']
        food_relation = inputs['food_relation']
        thorn_relation = inputs['thorn_relation']
        thorn_mask = inputs['thorn_mask']
        clone = inputs['clone']
        clone_relation = inputs['clone_relation']
        clone_mask = inputs['clone_mask']

        actor_embedding = self.actor_encoder(scalar, food, food_relation, thorn_relation, thorn_mask, clone,
                                             clone_relation, clone_mask)
        critic_embedding = self.critic_encoder(scalar, food, food_relation, thorn_relation, thorn_mask, clone,
                                             clone_relation, clone_mask)
        act = self.actor_head(actor_embedding)
        action_logit = act['logit']  # B, M, action_type_size
        action_type_logit = action_logit.reshape(B, A, *action_logit.shape[1:])

        value = self.critic_head(critic_embedding)
        value_pred = value['pred']
        value_type_pred = value_pred.reshape(B, A, *value_pred.shape[1:])
        value_output_pred = torch.mean(value_type_pred, 1)

        return {'logit': action_type_logit, 'value': value_output_pred}

