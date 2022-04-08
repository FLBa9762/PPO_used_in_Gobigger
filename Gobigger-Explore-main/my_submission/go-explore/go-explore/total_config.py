exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 5,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'reset',
            'retry_waiting_time': 0.1,
            'shared_memory': False,
            'context': 'spawn',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'force_reproducibility': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict'
        },
        'collector_env_num': 16,
        'evaluator_env_num': 3,
        'n_evaluator_episode': 3,
        'stop_value': 10000000000.0,
        'team_num': 4,
        'player_num_per_team': 3,
        'match_time': 600,
        'map_height': 1000,
        'map_width': 1000,
        'spatial': False,
        'speed': False,
        'all_vision': False,
        'reorder_team': True,
        'reorder_player': True,
        'frame_resume': True,
        'frame_path': './frame',
        'frame_cfg': {
            'type': 'dist'
        },
        'frame_period': 10
    },
    'policy': {
        'model': {
            'scalar_shape': 5,
            'food_shape': 2,
            'food_relation_shape': 150,
            'thorn_relation_shape': 12,
            'clone_shape': 17,
            'clone_relation_shape': 12,
            'hidden_shape': 128,
            'encode_shape': 32,
            'action_type_shape': 16
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 1000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 8,
            'batch_size': 256,
            'learning_rate': 0.001,
            'target_update_freq': 100,
            'ignore_done': False,
            'target_theta': 0.005,
            'discount_factor': 0.99
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'BattleSampleSerialCollectorDict'
            },
            'unroll_len': 1,
            'n_sample': 512,
            'alpha': 1.0
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'cfg_type': 'BattleInteractionSerialEvaluatorDict',
                'stop_value': 10000000000.0,
                'n_episode': 3
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'naive',
                'replay_buffer_size': 100000,
                'deepcopy': False,
                'enable_track_used_data': False,
                'periodic_thruput_seconds': 60,
                'cfg_type': 'NaiveReplayBufferDict'
            },
            'eps': {
                'type': 'exp',
                'start': 0.95,
                'end': 0.5,
                'decay': 100000
            }
        },
        'type': 'dqn',
        'cuda': True,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'discount_factor': 0.97,
        'nstep': 1,
        'cfg_type': 'DQNPolicyDict'
    },
    'exp_name': 'go-explore',
    'seed': 0
}
