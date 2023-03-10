class FlatTransition():
    def __init__(self, obs, action, reward, new_obs, done, info={}):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.new_obs = new_obs
        self.done = done
        self.info = info

    @classmethod
    def from_env_info(cls, obs, action, env_info):
        return cls(obs, action, env_info.reward, env_info.new_obs, env_info.done)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "obs: {}\n".format(self.obs) + \
        "action: {}\n".format(self.action) + \
        "reward: {}\n".format(self.reward) + \
        "new_obs: {}\n".format(self.new_obs) + \
        "done: {}\n".format(self.done) + \
        "info: {}\n".format(self.info)

