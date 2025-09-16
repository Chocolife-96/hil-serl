import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
# from pynput import keyboard
import sys, threading, atexit, termios, tty, select, time


from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


success_key = False

def _stdin_key_listener():
    """
    在独立线程中监听 SSH 终端的单键输入：
      空格 => success_key = True
      q    => 退出监听线程（可选）
    """
    global success_key
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    def restore():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except Exception:
            pass

    atexit.register(restore)
    try:
        tty.setraw(fd)  # raw 模式，单字符读取
        while True:
            # 使用 select 做非阻塞等待，避免占满 CPU
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if sys.stdin in rlist:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch == ' ':
                    success_key = True
                elif ch in ('q', '\x03', '\x04'):  # 'q' 或 Ctrl-C/Ctrl-D 退出监听
                    break
    finally:
        restore()

def start_key_listener():
    t = threading.Thread(target=_stdin_key_listener, daemon=True)
    t.start()
    return t

def main(_):
    global success_key
    # listener = keyboard.Listener(
    #     on_press=on_press)
    # listener.start()
    start_key_listener()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]() # ObjectHandoverTrainConfig()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    # obs, _ = env.reset()
    obs = None
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step_record(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                # actions=actions,
                # next_observations=next_obs,
                # rewards=rew,
                # masks=1.0 - done,
                # dones=done,
            )
        )
        
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
        else:
            if obs != None:
                failures.append(transition)
        obs = next_obs

        # if done or truncated:
        #     obs, _ = env.reset()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
