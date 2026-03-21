# config.py
import json
import os
import configparser
from pathlib import Path

class AssistantConfig:
    def __init__(self, config_path=None, env_prefix="ASSISTANT_"):
        # 默认配置
        self.name = "小灵"
        self.personality = "温柔细心"
        self.model = "qwen2.5:1.5b"
        self.ollama_url = "http://localhost:11434"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.memory_size = 50
        self.memory_file = None
        self.state_file = None
        self.enable_rules = True
        self.rag_enable = True
        self.knowledge_file = "test_rag.txt"
        self.rag_top_k = 3
        self.max_history = 10
        self.llm_num_ctx = 1024
        # config.py 中 __init__ 方法增加
        self.use_cloud_api = None
        self.cloud_api_key = "sk-zyyvofcbfeksdbokvktwyyhxjpnuxvaxwdtxedheswotgeqs"
        self.cloud_model = "Qwen/Qwen2.5-7B-Instruct"
        self.cloud_base_url = "https://api.siliconflow.cn/v1/chat/completions"  # 硅基流动会话

        self.env_prefix = env_prefix

        # 如果提供了配置文件，先加载
        if config_path:
            self._load_from_file(config_path)

        # 再加载环境变量（优先级更高）
        self._load_from_env()

        # 更新 role
        self.role = f"你是{self.name}，一个{self.personality}的本地助手。请用中文回答。"

    def _load_from_file(self, config_path):
        path = Path(config_path)
        if not path.exists():
            print(f"配置文件 {config_path} 不存在，跳过")
            return
        if path.suffix.lower() == '.ini':
            self._load_from_ini(config_path)
        elif path.suffix.lower() == '.json':
            self._load_from_json(config_path)
        else:
            print(f"不支持的配置文件格式: {path.suffix}")

    def _load_from_ini(self, ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')
        section = 'assistant' if config.has_section('assistant') else 'DEFAULT'
        for key, value in config[section].items():
            if hasattr(self, key):
                # 简单类型转换
                attr_type = type(getattr(self, key))
                try:
                    if attr_type == bool:
                        setattr(self, key, value.lower() in ('true', 'yes', '1'))
                    elif attr_type == int:
                        setattr(self, key, int(value))
                    elif attr_type == float:
                        setattr(self, key, float(value))
                    else:
                        setattr(self, key, value)
                except ValueError:
                    print(f"配置项 {key} 转换失败，保持默认值")

    def _load_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _load_from_env(self):
        """加载环境变量，格式：{PREFIX}{KEY}"""
        prefix = self.env_prefix.upper()
        for key in dir(self):
            if key.startswith('_') or key in ('env_prefix', 'role'):
                continue
            env_key = prefix + key.upper()
            if env_key in os.environ:
                env_value = os.environ[env_key]
                attr_type = type(getattr(self, key))
                try:
                    if attr_type == bool:
                        setattr(self, key, env_value.lower() in ('true', 'yes', '1'))
                    elif attr_type == int:
                        setattr(self, key, int(env_value))
                    elif attr_type == float:
                        setattr(self, key, float(env_value))
                    else:
                        setattr(self, key, env_value)
                except ValueError:
                    print(f"环境变量 {env_key} 转换失败")
