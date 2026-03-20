# tools.py
import datetime
import subprocess

def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间，可指定格式"""
    return datetime.datetime.now().strftime(format)

def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟）"""
    # 可以接入真实天气 API，目前此处返回模拟数据
    return f"{city} 的天气：晴，温度 22°C"

def calculate(expression: str) -> float:
    """计算数学表达式，如 '1+2*3'"""
    try:
        # 注意：eval 有安全风险，仅用于演示，生产环境请使用 safer eval 或 ast.literal_eval
        return eval(expression)
    except:
        return 0.0

def execute_safe_command(command: str) -> str:
    """安全执行系统命令（仅限白名单）"""
    allowed = ["ls", "pwd", "echo", "date"]
    parts = command.strip().split()
    if not parts or parts[0] not in allowed:
        return f"错误：命令 '{parts[0]}' 不在白名单中"
    result = subprocess.run(parts, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr

