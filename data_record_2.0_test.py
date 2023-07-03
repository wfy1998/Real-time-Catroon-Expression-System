from pynput import keyboard
recoding = False
stop = False
def on_press(key):
    global recoding, stop
    if key == keyboard.Key.space:
        print("Space key pressed!")
        recoding = True
    elif key == keyboard.Key.q:
        print("Q key pressed. Exiting...")
        # 停止监听器
        stop = True
def on_release(key):
    global recoding                     
    if key == keyboard.Key.space:
        recoding =False
        print("Space key released!")

                                                                      
# 创建键盘监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)

# 启动监听器
listener.start()

# 主程序持续运行的逻辑
while True:
    if recoding:
        print("recording")
    if stop:
        listener.stop()
        break