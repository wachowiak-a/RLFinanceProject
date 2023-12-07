from finrl.agents.stablebaselines3.models import DRLAgent


def train_a2c_agent(
        env,
        n_steps=10,
        ent_coef=0.005,
        learning_rate=0.0002,
        total_timesteps=50000
):
    agent = DRLAgent(env)
    model_a2c = agent.get_model(model_name="a2c", model_kwargs={
        'n_steps': n_steps,
        'ent_coef': ent_coef,
        'learning_rate': learning_rate
    })
    return agent.train_model(
        model=model_a2c,
        tb_log_name='a2c',
        total_timesteps=total_timesteps
    )
