python -m scripts.run.hpo_optuna \
--algo herald \
--env Drawbridge \
--objective_name env/success/test_proper_reward \
--enqueue \
--n_seeds 2 \
--n_processes 2 \
--n_trials 160 \
--n_steps 500000 \
--graph_params_to_optimize \
level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma \
level_params_list[1].subtask_spec_params.reward_normalization_coefficient \
--early_stopping \
herald_node_layer_0_subtask/train/subgoal_achieved 0.05 mean \
env/success/train 0.05 any \
--n_checkpoints 10     
