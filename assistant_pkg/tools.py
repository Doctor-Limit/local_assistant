# tools.py
import datetime
import subprocess
import shlex
#使用安全计算
from simpleeval import simple_eval

# 工具注册表
_TOOLS = {}

def register_tool(func):
    """装饰器：注册工具函数"""
    _TOOLS[func.__name__] = func
    return func

def get_registered_tools():
    """返回所有已注册的工具函数列表"""
    return list(_TOOLS.values())

# ========== 工具定义 ==========
@register_tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间，可指定格式"""
    return datetime.datetime.now().strftime(format)

@register_tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟）"""
    # 可接入真实天气 API，此处返回模拟数据
    return f"{city} 的天气：晴，温度 22°C"

@register_tool
def calculate(expression: str) -> float:
    """安全计算数学表达式，如 '1+2*3'"""
    if len(expression) > 100:
        return 0.0
    try:
        # 采用simple_eval ，安全且无需额外配置
        result = simple_eval(expression)
        return float(result)
    except Exception:
        return 0.0

@register_tool
def execute_safe_command(command: str) -> str:
    """
    安全执行系统命令（仅限白名单命令，并对参数进行严格过滤）
    """
    allowed_commands = {"ls", "pwd", "echo", "date"}
    # 使用 shlex.split 正确解析命令行（处理引号、转义）
    try:
        parts = shlex.split(command)
    except ValueError:
        return "错误：命令格式无效"
    if not parts:
        return "错误：空命令"
    if parts[0] not in allowed_commands:
        return f"错误：命令 '{parts[0]}' 不在白名单中"
    # 对每个参数进行安全校验：禁止包含危险字符
    dangerous_chars = set(';&|`$(){}[]<>')
    for arg in parts[1:]:
        if any(c in dangerous_chars for c in arg):
            return f"错误：参数中包含危险字符：{arg}"
    try:
        result = subprocess.run(parts, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"错误 ({result.returncode}): {result.stderr}"
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时"
    except Exception as e:
        return f"执行异常：{e}"