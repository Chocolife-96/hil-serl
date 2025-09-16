import sys, threading, atexit, termios, tty, select, time
from absl import app, flags
# from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 5, "Number of successful transitions to collect.")

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
    # 启动键盘监听（通过 SSH 终端）
    start_key_listener()

    # 你的原逻辑示例：轮询 success_key
    successes = 0
    print("监听中：在 SSH 终端按[空格]标记一次成功，按[q]结束监听线程。")
    while successes < FLAGS.successes_needed:
        if success_key:
            successes += 1
            # 读到一次就清零，等待下一次按键
            globals()['success_key'] = False
            print(f"[+] 收到一次成功标记，累计 {successes}/{FLAGS.successes_needed}")
        # ... 这里跑你的采集/训练逻辑 ...
        time.sleep(0.02)

    print("采集完成。")

if __name__ == "__main__":
    app.run(main)