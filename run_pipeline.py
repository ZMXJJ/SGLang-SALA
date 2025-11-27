#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化测试流水线 (Automated Test Pipeline)
功能：打包Python环境、解压到临时目录、生成测试代码、执行推理测试

用法:
    python run_pipeline.py [选项]
    
选项:
    --config FILE           指定配置文件路径 (默认: 优先查找当前目录config.sh，其次ExpAuto/config.sh)
    --no-cleanup            测试完成后不清理临时文件
    --model-path PATH       模型路径
    --env-type TYPE         环境类型 (venv/conda)
    --env-path PATH         venv路径或conda环境名
    --cuda-devices DEVICES  CUDA_VISIBLE_DEVICES
    --gpu-util FLOAT        GPU显存使用率 (0.0-1.0)
    -h, --help              显示帮助信息
"""

import os
import sys
import subprocess
import tarfile
import tempfile
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# ============================================================================
# 配置数据类
# ============================================================================
@dataclass
class PipelineConfig:
    """流水线配置"""
    # 模型配置
    model_path: str = "/path/to/your/model"
    test_prompt: str = "1+1=?"
    
    # GPU 配置
    gpu_memory_utilization: float = 0.9
    cuda_visible_devices: str = ""
    
    # 环境配置
    env_type: str = "venv"  # venv 或 conda
    venv_path: str = "./venv"
    conda_env_name: str = "base"
    
    # 临时目录配置
    temp_dir_prefix: str = "/tmp/test_pipeline"
    
    # 运行配置
    cleanup_after_test: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """从 YAML 文件加载配置"""
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        except ImportError:
            logging.warning("PyYAML 未安装，无法加载 YAML 配置")
            return cls()
        except FileNotFoundError:
            logging.warning(f"配置文件不存在: {yaml_path}，使用默认配置")
            return cls()
    
    @classmethod
    def from_shell_config(cls, config_path: str) -> "PipelineConfig":
        """从 Shell 配置文件加载配置（兼容原有 config.sh）"""
        config = cls()
        if not os.path.exists(config_path):
            logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return config
        
        logging.info(f"正在加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                
                # 解析 KEY=VALUE 或 KEY="VALUE"
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # 处理环境变量引用
                if value.startswith('${') and ':-' in value:
                    # ${VAR:-default} 格式
                    value = value.split(':-')[1].rstrip('}')
                
                # 映射到配置字段
                key_mapping = {
                    'MODEL_PATH': 'model_path',
                    'TEST_PROMPT': 'test_prompt',
                    'GPU_MEMORY_UTILIZATION': 'gpu_memory_utilization',
                    'CUDA_VISIBLE_DEVICES': 'cuda_visible_devices',
                    'ENV_TYPE': 'env_type',
                    'VENV_PATH': 'venv_path',
                    'CONDA_ENV_NAME': 'conda_env_name',
                    'TEMP_DIR_PREFIX': 'temp_dir_prefix',
                    'CLEANUP_AFTER_TEST': 'cleanup_after_test',
                }
                
                if key in key_mapping:
                    attr_name = key_mapping[key]
                    if attr_name == 'gpu_memory_utilization':
                        try:
                            value = float(value)
                        except ValueError:
                            logging.warning(f"无法解析 GPU_MEMORY_UTILIZATION: {value}，使用默认值")
                            continue
                    elif attr_name == 'cleanup_after_test':
                        value = value.lower() == 'true'
                    setattr(config, attr_name, value)
        
        return config


# ============================================================================
# 日志配置
# ============================================================================
class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;34m',     # Blue
        'WARNING': '\033[1;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'SUCCESS': '\033[0;32m',  # Green
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 添加自定义 SUCCESS 级别
        if record.levelno == 25:
            record.levelname = 'SUCCESS'
        
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}[{record.levelname}]{self.RESET}"
        return super().format(record)


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """设置日志"""
    # 添加自定义 SUCCESS 级别
    if not hasattr(logging, 'SUCCESS'):
        logging.SUCCESS = 25
        logging.addLevelName(logging.SUCCESS, 'SUCCESS')
    
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.SUCCESS):
            self._log(logging.SUCCESS, message, args, **kwargs)
    
    if not hasattr(logging.Logger, 'success'):
        logging.Logger.success = success
    
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    
    # 清除现有的 handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(levelname)s %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# 流水线类
# ============================================================================
class TestPipeline:
    """自动化测试流水线"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.temp_dir: Optional[Path] = None
        self.archive_path: Optional[Path] = None
        self.extracted_env_path: Optional[Path] = None
        
        # 设置日志
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f"pipeline_{timestamp}.log"
        self.logger = setup_logging(self.log_file)
    
    def run(self) -> int:
        """运行完整流水线"""
        try:
            self._print_header()
            
            # 步骤 1: 环境打包
            self._pack_environment()
            
            # 步骤 2: 环境解压
            self._extract_environment()
            
            # 步骤 3: 生成测试代码
            self._generate_test_code()
            
            # 步骤 4: 执行测试
            exit_code = self._run_test()
            
            self.logger.info("=" * 60)
            if exit_code == 0:
                self.logger.success("流水线执行完成!")
            else:
                self.logger.error(f"流水线执行失败，退出码: {exit_code}")
            self.logger.info("=" * 60)
            
            return exit_code
            
        except Exception as e:
            self.logger.error(f"流水线执行异常: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
        finally:
            self._cleanup()
    
    def _print_header(self):
        """打印启动信息"""
        self.logger.info("=" * 60)
        self.logger.info("自动化测试流水线启动")
        self.logger.info("=" * 60)
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(f"环境类型: {self.config.env_type}")
        self.logger.info(f"模型路径: {self.config.model_path}")
        self.logger.info(f"测试 Prompt: {self.config.test_prompt}")
        self.logger.info(f"CUDA_VISIBLE_DEVICES: {self.config.cuda_visible_devices or '未设置'}")
        self.logger.info(f"GPU 显存使用率: {self.config.gpu_memory_utilization}")
        self.logger.info(f"清理临时文件: {self.config.cleanup_after_test}")
        self.logger.info("")
    
    def _get_env_path(self) -> Path:
        """获取环境路径"""
        if self.config.env_type == "venv":
            env_path = Path(self.config.venv_path)
            if not env_path.exists():
                raise FileNotFoundError(f"venv 目录不存在: {env_path}")
            return env_path
            
        elif self.config.env_type == "conda":
            # 获取 conda 环境路径
            # 尝试多种方式获取 conda 路径
            try:
                # 方法 1: conda info --envs
                result = subprocess.run(
                    ["conda", "info", "--envs"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.startswith('#'): continue
                        parts = line.split()
                        if len(parts) >= 2 and parts[0] == self.config.conda_env_name:
                            env_path = Path(parts[-1])
                            if env_path.exists():
                                return env_path
            except Exception:
                pass

            # 方法 2: conda env list
            try:
                result = subprocess.run(
                    ["conda", "env", "list"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.startswith('#'): continue
                        # 匹配环境名，可能是 "name path" 或 "name * path"
                        parts = line.split()
                        if len(parts) >= 2 and parts[0] == self.config.conda_env_name:
                            env_path = Path(parts[-1])
                            if env_path.exists():
                                return env_path
            except Exception:
                pass
            
            raise FileNotFoundError(f"conda 环境不存在或无法获取路径: {self.config.conda_env_name}")
        
        else:
            raise ValueError(f"不支持的环境类型: {self.config.env_type}")
    
    def _pack_environment(self):
        """步骤 1: 环境打包"""
        self.logger.info("=" * 10 + " 步骤 1: 环境打包 " + "=" * 10)
        
        # 获取环境路径
        source_path = self._get_env_path()
        self.logger.info(f"检测到 {self.config.env_type} 环境: {source_path}")
        
        # 创建临时目录
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=os.path.basename(self.config.temp_dir_prefix) + "_"))
            self.logger.info(f"创建临时目录: {self.temp_dir}")
        except Exception as e:
            self.logger.error(f"无法创建临时目录: {e}")
            raise
        
        # 打包环境
        self.archive_path = self.temp_dir / "python_env_backup.tar.gz"
        self.logger.info(f"正在打包环境到: {self.archive_path}")
        self.logger.info(f"源目录: {source_path}")
        self.logger.info("这可能需要几分钟时间...")
        
        # 使用 tar 命令打包，以保持与 shell 脚本一致的行为（特别是符号链接处理）
        # 虽然 python tarfile 也能处理，但调用系统 tar 可能更稳健
        try:
            cmd = [
                "tar",
                "-czhf",
                str(self.archive_path),
                "-C",
                str(source_path.parent),
                source_path.name
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"打包失败: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError("打包失败")
        
        if not self.archive_path.exists():
            raise RuntimeError("打包失败，压缩包未生成")
        
        archive_size = self.archive_path.stat().st_size / (1024 ** 3)  # GB
        self.logger.success(f"环境打包完成，压缩包大小: {archive_size:.1f}G")
        self.logger.info("")
    
    def _extract_environment(self):
        """步骤 2: 环境解压"""
        self.logger.info("=" * 10 + " 步骤 2: 环境解压 " + "=" * 10)
        
        if not self.archive_path or not self.archive_path.exists():
            raise FileNotFoundError(f"压缩包不存在: {self.archive_path}")
        
        # 创建解压目标目录
        clean_env_dir = self.temp_dir / "clean_environment"
        clean_env_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"正在解压到纯净环境目录: {clean_env_dir}")
        
        # 解压
        try:
            cmd = [
                "tar",
                "-xzf",
                str(self.archive_path),
                "-C",
                str(clean_env_dir)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"解压失败: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError("解压失败")
        
        # 获取解压后的环境路径
        extracted_dirs = list(clean_env_dir.iterdir())
        if not extracted_dirs:
            raise RuntimeError("解压失败，目录为空")
        
        # 找到第一个目录作为解压后的环境路径
        self.extracted_env_path = extracted_dirs[0]
        if not self.extracted_env_path.is_dir():
             # 如果直接解压出来是文件而不是目录，可能结构不对，这里假设 tar 包结构是 目录/内容
             # 如果 tar 包里直接是内容，那 clean_env_dir 就是环境路径
             # 简单起见，假设结构正确，取第一个子项
             pass

        # 检查 Python 解释器
        python_bin = self.extracted_env_path / "bin" / "python"
        if not python_bin.exists():
            python_bin = self.extracted_env_path / "bin" / "python3"
        
        if not python_bin.exists():
            # 尝试在 clean_env_dir 直接查找
            python_bin = clean_env_dir / "bin" / "python"
            if python_bin.exists():
                self.extracted_env_path = clean_env_dir
            else:
                 # 再次尝试更深层级搜索（防止多层嵌套）
                 pass
        
        if not python_bin.exists():
            raise FileNotFoundError(f"Python 解释器未找到: {self.extracted_env_path}/bin/python")
        
        self.logger.success("环境解压完成")
        self.logger.info(f"Python 解释器路径: {python_bin}")
        
        # 显示 Python 版本
        result = subprocess.run([str(python_bin), "--version"], capture_output=True, text=True)
        self.logger.info(f"Python 版本信息: {result.stdout.strip() or result.stderr.strip()}")
        self.logger.info("")
    
    def _generate_test_code(self):
        """步骤 3: 生成测试代码"""
        self.logger.info("=" * 10 + " 步骤 3: 生成测试代码 " + "=" * 10)
        
        test_file = self.temp_dir / "test_inference.py"
        self.logger.info(f"正在生成测试文件: {test_file}")
        
        # 测试代码模板 (保持原样)
        test_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化推理测试脚本
使用 vLLM 库加载模型并执行简单推理测试
"""

import sys
import os
import time
from typing import List

def main():
    print("=" * 60)
    print("vLLM 推理测试")
    print("=" * 60)
    
    # 从环境变量获取配置
    model_path = os.environ.get("MODEL_PATH", "/path/to/your/model")
    test_prompt = os.environ.get("TEST_PROMPT", "1+1=?")
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
    
    print(f"\\n[配置信息]")
    print(f"  模型路径: {model_path}")
    print(f"  测试 Prompt: {test_prompt}")
    print(f"  GPU 显存使用率: {gpu_memory_utilization}")
    print(f"  Python 版本: {sys.version}")
    print(f"  Python 路径: {sys.executable}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"\\n[警告] 模型路径不存在: {model_path}")
        print("请设置正确的 MODEL_PATH")
        print("当前将使用演示模式运行...")
        run_demo_mode(test_prompt)
        return
    
    try:
        print("\\n[步骤 1] 导入 vLLM 库...")
        start_time = time.time()
        from vllm import LLM, SamplingParams
        import_time = time.time() - start_time
        print(f"  vLLM 导入成功 (耗时: {import_time:.2f}s)")
        
        print("\\n[步骤 2] 加载模型...")
        start_time = time.time()
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization
        )
        load_time = time.time() - start_time
        print(f"  模型加载成功 (耗时: {load_time:.2f}s)")
        
        print("\\n[步骤 3] 设置采样参数...")
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        print(f"  temperature: {sampling_params.temperature}")
        print(f"  top_p: {sampling_params.top_p}")
        print(f"  max_tokens: {sampling_params.max_tokens}")
        
        print("\\n[步骤 4] 执行推理...")
        prompts: List[str] = [test_prompt]
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        inference_time = time.time() - start_time
        print(f"  推理完成 (耗时: {inference_time:.2f}s)")
        
        print("\\n" + "=" * 60)
        print("推理结果")
        print("=" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\\n[Prompt]: {prompt}")
            print(f"[Output]: {generated_text}")
        
        print("\\n" + "=" * 60)
        print("测试完成 - 成功")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\\n[错误] vLLM 库导入失败: {e}")
        print("请确保 vLLM 已正确安装在当前环境中")
        print("安装命令: pip install vllm")
        sys.exit(1)
        
    except Exception as e:
        print(f"\\n[错误] 推理过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_demo_mode(test_prompt: str):
    """演示模式"""
    print("\\n" + "=" * 60)
    print("演示模式")
    print("=" * 60)
    
    try:
        print("\\n[步骤 1] 检查 vLLM 库是否可用...")
        import vllm
        print(f"  vLLM 版本: {vllm.__version__}")
        print("  vLLM 库可用!")
        
        print("\\n[步骤 2] 显示演示输出...")
        print(f"\\n[Prompt]: {test_prompt}")
        print(f"[Demo Output]: 2 (这是演示输出，实际推理需要配置有效的模型路径)")
        
        print("\\n" + "=" * 60)
        print("演示模式完成 - vLLM 环境验证成功")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\\n[错误] vLLM 库不可用: {e}")
        print("请安装 vLLM: pip install vllm")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        os.chmod(test_file, 0o755)
        
        self.logger.success(f"测试文件生成完成: {test_file}")
        self.logger.info("测试文件内容预览:")
        
        # 预览前 20 行
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 20:
                    self.logger.info("...")
                    break
                print(line.rstrip())
        
        self.logger.info("")
    
    def _run_test(self) -> int:
        """步骤 4: 执行测试"""
        self.logger.info("=" * 10 + " 步骤 4: 执行测试 " + "=" * 10)
        
        test_file = self.temp_dir / "test_inference.py"
        
        # 获取 Python 解释器路径
        # 重新定位解释器（以防万一）
        python_bin = self.extracted_env_path / "bin" / "python"
        if not python_bin.exists():
             python_bin = self.extracted_env_path / "bin" / "python3"
        
        if not python_bin.exists():
            # Fallback search
            for root, dirs, files in os.walk(self.extracted_env_path):
                if "python" in files:
                    python_bin = Path(root) / "python"
                    break
        
        if not python_bin or not python_bin.exists():
            raise FileNotFoundError("Python 解释器未找到")
        
        if not test_file.exists():
            raise FileNotFoundError(f"测试文件未找到: {test_file}")
        
        self.logger.info(f"使用 Python 解释器: {python_bin}")
        self.logger.info(f"执行测试文件: {test_file}")
        self.logger.info(f"模型路径: {self.config.model_path}")
        self.logger.info(f"测试 Prompt: {self.config.test_prompt}")
        self.logger.info(f"GPU 显存使用率: {self.config.gpu_memory_utilization}")
        
        if self.config.cuda_visible_devices:
            self.logger.info(f"CUDA_VISIBLE_DEVICES: {self.config.cuda_visible_devices}")
        
        self.logger.info("")
        self.logger.info("=" * 10 + " 测试输出开始 " + "=" * 10)
        self.logger.info("")
        
        # 设置环境变量
        env = os.environ.copy()
        env["MODEL_PATH"] = self.config.model_path
        env["TEST_PROMPT"] = self.config.test_prompt
        env["GPU_MEMORY_UTILIZATION"] = str(self.config.gpu_memory_utilization)
        
        if self.config.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
            self.logger.info(f"CUDA_VISIBLE_DEVICES: {self.config.cuda_visible_devices}")
        
        # 运行测试
        # 确保当前目录是 temp_dir，这样生成的任何文件都在那里
        try:
            result = subprocess.run(
                [str(python_bin), str(test_file)],
                env=env,
                cwd=str(self.temp_dir),
                check=False 
            )
        except Exception as e:
            self.logger.error(f"运行测试脚本失败: {e}")
            return 1
        
        self.logger.info("")
        self.logger.info("=" * 10 + " 测试输出结束 " + "=" * 10)
        self.logger.info("")
        
        if result.returncode == 0:
            self.logger.success(f"测试执行完成，退出码: {result.returncode}")
        else:
            self.logger.error(f"测试执行失败，退出码: {result.returncode}")
        
        return result.returncode
    
    def _cleanup(self):
        """清理临时文件"""
        if self.config.cleanup_after_test and self.temp_dir and self.temp_dir.exists():
            self.logger.info(f"正在清理临时目录: {self.temp_dir}")
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.success("临时目录清理完成")
            except Exception as e:
                self.logger.warning(f"清理临时目录失败: {e}")
        elif self.temp_dir:
            self.logger.info(f"临时目录保留在: {self.temp_dir}")
            self.logger.info(f"测试文件: {self.temp_dir}/test_inference.py")
            if self.archive_path:
                self.logger.info(f"压缩包: {self.archive_path}")


# ============================================================================
# 命令行入口
# ============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="自动化测试流水线 - 打包Python环境并测试vLLM推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_pipeline.py
    python run_pipeline.py --no-cleanup
    python run_pipeline.py --model-path /path/to/model --cuda-devices 0,1
    python run_pipeline.py --config ExpAuto/config.sh
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="配置文件路径 (支持 .sh 或 .yaml 格式)"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="测试完成后不清理临时文件"
    )
    
    parser.add_argument(
        "--model-path",
        help="模型路径"
    )
    
    parser.add_argument(
        "--env-type",
        choices=["venv", "conda"],
        help="环境类型"
    )
    
    parser.add_argument(
        "--env-path",
        help="venv路径或conda环境名"
    )
    
    parser.add_argument(
        "--cuda-devices",
        help="CUDA_VISIBLE_DEVICES"
    )
    
    parser.add_argument(
        "--gpu-util",
        type=float,
        help="GPU显存使用率 (0.0-1.0)"
    )
    
    parser.add_argument(
        "--test-prompt",
        help="测试提示词"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 确定配置文件路径
    config_path_str = args.config
    if not config_path_str:
        # 默认查找顺序: ./config.sh -> ExpAuto/config.sh
        if os.path.exists("config.sh"):
            config_path_str = "config.sh"
        elif os.path.exists("ExpAuto/config.sh"):
            config_path_str = "ExpAuto/config.sh"
        else:
            config_path_str = "config.sh" # Fallback
    
    config_path = Path(config_path_str)
    
    # 加载配置
    if config_path.suffix in ['.yaml', '.yml']:
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        # 默认尝试加载 shell 配置文件
        config = PipelineConfig.from_shell_config(str(config_path))
    
    # 命令行参数覆盖配置文件
    if args.no_cleanup:
        config.cleanup_after_test = False
    
    if args.model_path:
        config.model_path = args.model_path
    
    if args.env_type:
        config.env_type = args.env_type
    
    if args.env_path:
        if config.env_type == "venv":
            config.venv_path = args.env_path
        else:
            config.conda_env_name = args.env_path
    
    if args.cuda_devices:
        config.cuda_visible_devices = args.cuda_devices
    
    if args.gpu_util is not None:
        config.gpu_memory_utilization = args.gpu_util
    
    if args.test_prompt:
        config.test_prompt = args.test_prompt
    
    # 运行流水线
    pipeline = TestPipeline(config)
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

