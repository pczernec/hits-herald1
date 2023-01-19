import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(name="reacting_rl_envs",
      version="0.1.0",
      description="RL environments that require the agents to be able to react to changes",
      long_description=README,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/lychanl/reacting-rl-envs",
      keywords=["reinforcement learning", "reinforcement learning environmnts"],
      packages=find_packages(),
      install_requires=["numpy", "gym", "Box2D"]
)
