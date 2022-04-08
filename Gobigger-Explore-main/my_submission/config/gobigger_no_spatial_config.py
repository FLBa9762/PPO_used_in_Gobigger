from easydict import EasyDict

gobigger_config = dict(
    exp_name='gobigger_baseline_v030',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=3, 
        n_evaluator_episode=3,
        stop_value=1e10,
        team_num=4,
        player_num_per_team=1,
        match_time=60*10,
        map_height=1000,
        map_width=1000,
        spatial=False,
        speed = False,
        all_vision = False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        on_policy=True,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            scalar_shape=5,
            food_shape=2,
            food_relation_shape=150,
            thorn_relation_shape=12,
            clone_shape=17,
            clone_relation_shape=12,
            hidden_shape=128,
            encode_shape=32,
            action_type_shape=16,
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=128,
            learning_rate=1e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            # entropy_weight=0.01,
            clip_ratio=0.2,
            ignore_done=False,
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000)),
        ),
        collect=dict(n_sample=1024, unroll_len=1, discount_factor=0.9, gae_lambda=0.95),
        eval=dict(evaluator=dict(eval_freq=1000,)),
        continuous=False,
    ),
)

main_config = EasyDict(gobigger_config)
gobigger_create_config = dict(
    env=dict(
        type='gobigger',
        import_names=['dizoo.gobigger.envs.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(gobigger_create_config)
