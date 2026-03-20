# main.py
from assistant_pkg.assistant import Assistant

def main():
    # 可以传入配置文件路径，此处使用默认配置
    assistant = Assistant()  # 或者 Assistant("config.json")
    print(f"{assistant.config.name} 已启动，输入 'exit' 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['exit', 'quit']:
            print("再见！")
            break
        response_generator = assistant.process_stream(user_input)
        for chunk in response_generator:
            # 每收到一块，立即处理（例如打印）
            safe_chunk = chunk.encode('utf-8', 'ignore').decode('utf-8')
            print(safe_chunk, end='', flush=True)  # 不换行打印
        print()  # 最后换行

if __name__ == "__main__":
    main()

