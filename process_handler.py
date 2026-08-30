# process_handler.py

import cv2
import yaml
import time
import importlib
import threading
import traceback
import numpy as np
from pynput import keyboard
from pynput.mouse import Button
from log import log
from actions import ActionExecutor
from reward_calculator import RewardJudge


class TrainingManager:
    def __init__(self, context, running_event, osd_event=None):
        self.context = context
        self.running_event = running_event
        self.osd_event = osd_event  # OSD 叠加层开关事件
        self.context.reopen_shared_memory()

        # 初始化队列
        self.emergency_queue = context.get_emergency_event_queue()
        self.normal_queue = context.get_normal_event_queue()

        # 加载配置
        self.config = self._load_config()
        self.training_config = self.config["training"]
        self.env_config = self.config["environment"]

        # 初始化组件
        self.executor = ActionExecutor("./config/actions_conf.yaml")
        self.judger = RewardJudge()

        # 设置维度
        self.context_dim = context.get_features_len()
        self.image_state_dim = (self.env_config["height"], self.env_config["width"])
        # v2: 传入图像维度元组给 SAC，让 CNN 处理（不再 flatten）
        self.state_dim = (self.env_config["height"], self.env_config["width"])
        self.action_dim = self.executor.get_action_size()

        # 初始化智能体
        self.agent = self._initialize_agent()

        # 训练控制
        self.training_mode = threading.Event()
        self.training_mode.clear()
        self.episode = 0
        self.save_step = self.training_config["save_step"]

        # 设置键盘监听
        self.listener = self._setup_keyboard_listener()

    def _load_config(self):
        """加载配置文件"""
        with open("./config/models_conf.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _initialize_agent(self):
        """初始化Agent"""
        model_type = self.config["model"]["type"]
        model_config = self.config["model"].get(model_type, {})
        model_file = self.config["model"]["model_file"]

        log.info("正在加载模型中...")
        model_module = f"models.{model_type.lower()}"
        Agent = getattr(importlib.import_module(model_module), model_type)

        agent = Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            context_dim=self.context_dim,
            config=model_config,
            model_file=model_file,
        )
        log.debug("Agent创建完成!")
        return agent

    def _setup_keyboard_listener(self):
        """设置键盘监听器"""

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == "g":
                    if self.training_mode.is_set():
                        log.debug("暂停训练模式...")
                        log.info("等待按 g 开始训练")
                        self.executor.interrupt_action()
                        self.training_mode.clear()
                    else:
                        log.info("开始训练模式...")
                        self.training_mode.set()
            except AttributeError:
                pass

            # F1 切换 OSD 叠加层显示/隐藏（进程常驻，只切换可见性，响应即时）
            if key == keyboard.Key.f1 and self.osd_event is not None:
                if self.osd_event.is_set():
                    self.osd_event.clear()
                    log.info("🎮 OSD 叠加层已隐藏 (F1)")
                else:
                    self.osd_event.set()
                    log.info("🎮 OSD 叠加层已显示 (F1)")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        log.info("等待按 g 开始训练 | F1 切换 OSD 叠加层")
        return listener

    def get_current_state(self):
        """获取当前状态"""
        frame, status = self.context.get_frame_and_status()
        state_image = cv2.resize(frame[:, :, :3], self.image_state_dim[::-1])
        state_image_array = np.array(state_image).transpose(2, 0, 1)[
            np.newaxis, :, :, :
        ]
        context_features = self.context.get_features(status)
        return (state_image_array, context_features), status

    def clear_event_queues(self):
        """清空事件队列"""
        while not self.emergency_queue.empty():
            self.emergency_queue.get_nowait()
        while not self.normal_queue.empty():
            self.normal_queue.get_nowait()

    def handle_action_execution(self, action, action_name, start_time):
        """处理动作执行和事件监控"""
        events = []
        injured = False
        can_interrupt = self.executor.is_interruptible(action_name)
        interrupt_action_done = False
        action_start_time = time.time()
        done = False

        while self.executor.is_running():
            time.sleep(0.001)
            action_duration = time.time() - action_start_time

            # 处理紧急事件
            done, new_injured = self._handle_emergency_events(
                events,
                start_time,
                action_start_time,
                can_interrupt,
                interrupt_action_done,
            )

            injured = injured or new_injured

            if done:
                break

            # 处理普通事件
            self._handle_normal_events(events, start_time)

        return events, injured, action_duration, done

    def _handle_emergency_events(
        self,
        events,
        start_time,
        action_start_time,
        can_interrupt,
        interrupt_action_done,
    ):
        """处理紧急事件"""
        injured = False
        while not self.emergency_queue.empty():
            e_event = self.emergency_queue.get_nowait()
            if e_event["timestamp"] <= start_time:
                continue

            events.append(e_event)

            # 处理q_found事件
            if e_event["event"] == "q_found" and e_event["current_value"] == 0:
                done = self._handle_q_found_event(interrupt_action_done)
                if done:
                    return True, injured

            # 处理受伤事件
            if (
                e_event["event"] == "self_blood"
                and e_event["relative_change"] < -1.0
                and e_event["timestamp"] > action_start_time
            ):
                injured = True
                if not interrupt_action_done and can_interrupt:
                    self.executor.interrupt_action()
                    interrupt_action_done = True

        return False, injured

    def _handle_normal_events(self, events, start_time):
        """处理普通事件"""
        while not self.normal_queue.empty():
            n_event = self.normal_queue.get_nowait()
            if n_event["timestamp"] > start_time:
                events.append(n_event)

    def _handle_q_found_event(self, interrupt_action_done):
        """处理q_found事件"""
        q_found_time = time.time()

        # 添加日志以便调试
        log.debug(f"检测到q_found事件为0，开始等待确认状态")

        while time.time() - q_found_time < 0.5:
            time.sleep(0.01)
            if not self.emergency_queue.empty():
                delayed_event = self.emergency_queue.get_nowait()
                if (
                    delayed_event["event"] == "q_found"
                    and delayed_event["current_value"] == 1
                ):
                    log.debug("q_found恢复为1，忽略之前的事件")
                    return False

        # 添加额外的血量检查
        _, current_status = self.get_current_state()
        if current_status["self_blood"] > 0:
            log.debug("q_found为0但玩家血量正常，忽略q_found事件")
            return False

        log.debug("未检测到q_found恢复，且玩家血量为0，确认回合结束")
        if not interrupt_action_done:
            self.executor.interrupt_action()
        return True

    def handle_episode_restart(self):
        """处理重启逻辑"""
        if not self.training_mode.is_set():
            return

        log.debug("准备重新进入训练!")

        restart_load_time = 10
        log.debug(f"死亡加载动画等待中... 需{restart_load_time}秒")
        time.sleep(restart_load_time)

        restart_action_name = self.training_config["restart_action"]
        self.executor.take_action(restart_action_name)
        self.executor.wait_for_finish()

        log.debug(f"重新进入训练逻辑 结束.")

    def run(self):
        """主运行循环"""
        try:
            while self.running_event.is_set():
                if self.training_mode.is_set():
                    self._run_training_episode()
                else:
                    self.clear_event_queues()
                    time.sleep(0.03)
        except KeyboardInterrupt:
            log.error("进程: 正在退出...")
            self.running_event.clear()
        except Exception as e:
            error_message = traceback.format_exc()
            log.error(f"发生错误: {e}\n{error_message}")
            self.running_event.clear()
        finally:
            # 停止键盘监听器（加超时保护，防止阻塞）
            try:
                stop_thread = threading.Thread(target=self.listener.stop, daemon=True)
                stop_thread.start()
                stop_thread.join(timeout=2)
                if stop_thread.is_alive():
                    log.debug("键盘监听器停止超时，跳过")
            except Exception:
                pass

            # 关闭共享内存（加超时保护）
            try:
                close_thread = threading.Thread(target=self.context.close, daemon=True)
                close_thread.start()
                close_thread.join(timeout=3)
                if close_thread.is_alive():
                    log.debug("共享内存关闭超时，跳过")
            except Exception:
                pass

            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _run_training_episode(self):
        """运行训练回合"""
        total_episodes = self.training_config["episodes"]

        while self.episode < total_episodes and self.training_mode.is_set():
            log.info(f"第 {self.episode} 回合开始")

            # 重置评判器
            self.judger.reset()
            self.clear_event_queues()

            # 等待游戏正确加载
            log.info("等待游戏加载完成...")
            game_ready = False
            wait_start_time = time.time()
            while not game_ready and time.time() - wait_start_time < 60:  # 最多等待60秒
                _, status = self.get_current_state()
                if status["self_blood"] > 50 and status["boss_blood"] > 50:
                    game_ready = True
                    log.info("游戏已准备就绪！血量正常，开始训练")
                    break
                time.sleep(0.5)

            if not game_ready:
                log.error("等待超时，游戏可能未正确加载。尝试重新开始...")
                self.handle_episode_restart()
                continue

            # 开局按鼠标中键锁定Boss
            log.info("按鼠标中键锁定Boss...")
            self.executor.mouse.click(Button.middle)
            time.sleep(0.3)

            # 获取初始状态
            state, status = self.get_current_state()
            target_step = 0
            done = False
            start_time = time.time()
            injured_cnt = 0
            dodge_cnt = 0
            episode_reward = 0  # 回合总奖励

            # === 本回合详细统计数据（供 OSD 监控面板使用） ===
            ATTACK_ACTIONS = {'LIGHT_ATTACK', 'HEAVY_ATTACK', 'ATTACK_DODGE',
                              'FIVE_HIT_COMBO', 'QIESHOU', 'STEALTH_CHARGE'}
            SKILL_ACTIONS = {'SKILL_1', 'SKILL_2', 'SKILL_3', 'SKILL_4',
                             'TISHEN', 'FABAO', 'STEALTH_CHARGE'}
            DODGE_ACTIONS = {'DODGE', 'DODGE_TWO', 'DODGE_THREE'}

            ep_action_counts = {}     # 各动作使用次数
            ep_attack_count = 0       # AI 总攻击次数
            ep_boss_hit_count = 0     # Boss 被命中次数（血量下降事件）
            ep_skill_usage = {}       # 技能使用次数
            ep_dodge_total = 0        # 总闪避次数
            ep_dodge_success = 0      # 成功闪避（闪避后未受伤）
            ep_reward_history = []    # 每步奖励历史（最近120步）
            ep_self_blood_history = []   # 玩家血量历史（最近120步）
            ep_boss_blood_history = []   # Boss血量历史（最近120步）
            ep_damage_dealt = 0.0     # 对Boss造成的总伤害
            ep_damage_taken = 0.0     # 自身受到的总伤害
            HISTORY_MAX = 120         # 历史列表最大长度

            while not done and self.training_mode.is_set():
                # 选择动作
                action, log_prob = self.agent.choose_action(state)
                action_name = self.executor.get_action_name(action)

                # v3: 不再做CD硬拦截，agent自由选择动作
                # 如果选了CD中的技能，游戏本身会忽略输入（自然负反馈）
                log.debug(f"智能体采取 {action_name} 动作.")
                self.executor.take_action(action)

                # 执行动作并获取结果
                events, injured, action_duration, done = self.handle_action_execution(
                    action, action_name, start_time
                )

                if injured:
                    injured_cnt += 1
                    dodge_cnt = 0
                else:
                    dodge_cnt += 1  # 未受伤时递增闪避计数

                # --- 统计动作分布 ---
                ep_action_counts[action_name] = ep_action_counts.get(action_name, 0) + 1
                if action_name in ATTACK_ACTIONS:
                    ep_attack_count += 1
                if action_name in SKILL_ACTIONS:
                    ep_skill_usage[action_name] = ep_skill_usage.get(action_name, 0) + 1
                if action_name in DODGE_ACTIONS:
                    ep_dodge_total += 1
                    if not injured:
                        ep_dodge_success += 1

                # 记录动作结果
                if injured and self.executor.is_interruptible(action_name):
                    log.debug(
                        f"受伤了 {action_name} 动作提前结束 {action_duration:.2f}s."
                    )
                elif injured and not self.executor.is_interruptible(action_name):
                    log.debug(
                        f"{action_name} 动作不可中断 耗时 {action_duration:.2f}s."
                    )
                else:
                    log.debug(f"{action_name} 动作结束 {action_duration:.2f}s.")

                # 获取下一个状态
                next_state, next_status = self.get_current_state()

                # 检查游戏状态是否仍然有效
                if next_status["self_blood"] <= 0 and next_status["boss_blood"] <= 0:
                    # 双方血量都为0，通常是玩家死亡后死亡动画期间画面识别异常
                    # 用上一帧的Boss血量来还原真实状态，视为玩家死亡（失败）
                    log.info(
                        f"双方血量都为0（可能死亡动画中），"
                        f"使用上一帧Boss血量({status['boss_blood']:.2f})作为真实值，判定为玩家死亡"
                    )
                    next_status["boss_blood"] = status["boss_blood"]
                    next_status["self_blood"] = 0
                    injured = True
                    injured_cnt += 1
                    done = True
                    # 不要 continue，让后续的奖励计算正常执行（给予失败惩罚）

                # --- 检测回合是否结束（在 judge 之前判定，保证 done 状态正确传入） ---
                # done_by_event: handle_action_execution 已经通过事件设置了 done
                # done_by_blood: 本步骤通过血量检测新发现的结束
                done_by_blood = False
                if not done:
                    # 判断玩家失败：当前玩家血量小于1且上一帧Boss血量大于0
                    if next_status["self_blood"] < 1 and status["boss_blood"] > 0:
                        log.info("玩家失败！玩家血量小于1，上一帧Boss血量大于0")
                        done = True
                        done_by_blood = True

                    # 判断玩家胜利：当前Boss血量为0且玩家血量大于0
                    if next_status["boss_blood"] <= 0 and next_status["self_blood"] > 0:
                        log.info("玩家胜利！成功击败Boss")
                        done = True
                        done_by_blood = True

                # 计算奖励（done 状态已正确设置，judge 内部会在 done=True 时调用 end_episode）
                # 如果回合结束，构建详细战斗统计传给 judge → end_episode → CSV
                final_episode_stats = None
                if done:
                    final_episode_stats = {
                        'attack_count': ep_attack_count,
                        'boss_hit_count': ep_boss_hit_count,
                        'dodge_total': ep_dodge_total,
                        'dodge_success': ep_dodge_success,
                        'skill_use_count': sum(ep_skill_usage.values()),
                        'damage_dealt': round(ep_damage_dealt, 2),
                        'damage_taken': round(ep_damage_taken, 2),
                        'action_diversity': len(ep_action_counts),
                    }

                reward = self.judger.judge(
                    action_name,
                    injured,
                    status,
                    next_status,
                    events,
                    time.time() - start_time,
                    done,
                    injured_cnt,
                    steps=target_step,
                    episode_stats=final_episode_stats,
                )
                episode_reward += reward

                # --- 统计 Boss 被命中次数 & 伤害 ---
                boss_blood_drop = float(status["boss_blood"]) - float(next_status["boss_blood"])
                if boss_blood_drop > 0.5:  # Boss 血量明显下降才算命中
                    ep_boss_hit_count += 1
                    ep_damage_dealt += boss_blood_drop
                self_blood_drop = float(status["self_blood"]) - float(next_status["self_blood"])
                if self_blood_drop > 0:
                    ep_damage_taken += self_blood_drop

                # --- 记录历史曲线数据 ---
                ep_reward_history.append(round(reward, 2))
                ep_self_blood_history.append(round(float(next_status["self_blood"]), 1))
                ep_boss_blood_history.append(round(float(next_status["boss_blood"]), 1))
                if len(ep_reward_history) > HISTORY_MAX:
                    ep_reward_history.pop(0)
                if len(ep_self_blood_history) > HISTORY_MAX:
                    ep_self_blood_history.pop(0)
                if len(ep_boss_blood_history) > HISTORY_MAX:
                    ep_boss_blood_history.pop(0)

                # 构建本回合详细统计
                cur_duration = time.time() - start_time
                episode_stats = {
                    'action_counts': {k: int(v) for k, v in ep_action_counts.items()},
                    'attack_count': int(ep_attack_count),
                    'boss_hit_count': int(ep_boss_hit_count),
                    'skill_usage': {k: int(v) for k, v in ep_skill_usage.items()},
                    'dodge_count': int(ep_dodge_success),
                    'dodge_total': int(ep_dodge_total),
                    'dodge_success_rate': round(float(ep_dodge_success / ep_dodge_total), 2) if ep_dodge_total > 0 else 0.0,
                    'reward_history': [float(v) for v in ep_reward_history],
                    'self_blood_history': [float(v) for v in ep_self_blood_history],
                    'boss_blood_history': [float(v) for v in ep_boss_blood_history],
                    'blood_damage_dealt': round(float(ep_damage_dealt), 1),
                    'blood_damage_taken': round(float(ep_damage_taken), 1),
                    'avg_reward_per_step': round(float(episode_reward) / max(target_step, 1), 2),
                    'dps': round(float(ep_damage_dealt) / max(cur_duration, 0.1), 2),
                }

                # 更新实时状态（供 OSD 叠加层读取）
                self.judger.reward_tracker.update_realtime_state(
                    episode=self.episode,
                    step=target_step,
                    action_name=action_name,
                    reward=float(reward),
                    self_blood=float(next_status["self_blood"]),
                    boss_blood=float(next_status["boss_blood"]),
                    injured_count=int(injured_cnt),
                    episode_reward=float(episode_reward),
                    duration=time.time() - start_time,
                    epsilon=float(getattr(self.agent, 'epsilon', 0.0)),
                    ui_status={
                        'self_magic': round(float(next_status["self_magic"]), 1),
                        'self_energy': round(float(next_status["self_energy"]), 1),
                        'hulu': round(float(next_status["hulu"]), 1),
                        'skill_1': float(next_status["skill_1"]),
                        'skill_2': float(next_status["skill_2"]),
                        'skill_3': float(next_status["skill_3"]),
                        'skill_4': float(next_status["skill_4"]),
                        'skill_ts': float(next_status["skill_ts"]),
                        'skill_fb': float(next_status["skill_fb"]),
                        'gunshi1': float(next_status["gunshi1"]),
                        'gunshi2': float(next_status["gunshi2"]),
                        'gunshi3': float(next_status["gunshi3"]),
                    },
                    episode_stats=episode_stats,
                )

                # 存储转换并训练
                self.agent.store_data(
                    state, action, reward, next_state, done, log_prob, target_step
                )
                self.agent.train_network()

                target_step += 1
                if target_step % self.training_config["update_step"] == 0:
                    self.agent.update_target_network()

                state = next_state
                status = next_status.copy()

            if self.training_mode.is_set():
                self.episode += 1

                # 每 save_step 回合保存一次模型
                if self.episode % self.save_step == 0:
                    self.agent.save_model()
                    log.debug(f"模型在第 {self.episode} 回合保存")

                # 跳过 CG
                if self.training_mode.is_set():
                    self.executor.take_action("SKIP_CG")
                    self.executor.wait_for_finish()
                    log.debug("跳过 CG 完成!")

                # 处理重启
                self.handle_episode_restart()

            if self.episode >= total_episodes:
                log.debug("训练完成.")
                self.training_mode.clear()


def process(context, running_event, osd_event=None):
    """主入口函数"""
    manager = TrainingManager(context, running_event, osd_event)
    manager.run()
