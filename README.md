# PPO_used_in_Gobigger
使用了ppo算法实现了gobigger的单智能体训练模型，多智能体方法未实现。
安装DI版本为DI_engine-0.2.2

需要修改库
修改位置：
** 1： *****\envs\<>\Lib\site-packages\DI_engine-0.2.2-py3.8.egg\ding\utils\default_helper.py **
    split_data_generator()
    修改内容：460行-464行，修改 elif isinstance(v, dict)判断为字典类型后的运算，增加函数：gobigger_data_deal(dict_data)
```python
            elif isinstance(data[k], dict):
                gobigger_data, gobigger_batch, gobigger_player_num_per_team = gobigger_data_deal(data[k])
                learn_batch = {k1: v1[indices[i:i + split_size]] for k1, v1 in gobigger_data.items()}
                learn_batch['batch'] = split_size
                learn_batch['player_num_per_team'] = gobigger_player_num_per_team
                batch[k] = learn_batch
                # batch[k] = {k1: v1[indices[i:i + split_size]] for k1, v1 in data[k].items()}
            else:
                batch[k] = data[k][indices[i:i + split_size]]
        yield batch

def gobigger_data_deal(dict_data):  # 拆分出 batch 与 player_num_per_team
    assert isinstance(dict_data, dict), '修改有误，obs和next_obs不都为dict类型{}'.format(type(a))
    control_data = copy.deepcopy(dict_data)
    gobigger_batch = control_data['batch']
    gobigger_player_num_per_team = control_data['player_num_per_team']
    del control_data['batch']
    del control_data['player_num_per_team']
    return control_data, gobigger_batch, gobigger_player_num_per_team
```
