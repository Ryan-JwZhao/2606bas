from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ..operator_controls import normalize_shot_mode
from ..schemas import to_jsonable


class WebControlOperatorMixin:
    @QtCore.pyqtSlot()
    def toggle_web_control(self) -> None:
        if self.web_control.is_running:
            self.stop_web_control()
        else:
            self.start_web_control()

    def start_web_control(self) -> bool:
        if self.web_control.is_running:
            return True
        host = str(self.config.web_control.host or "0.0.0.0").strip() or "0.0.0.0"
        port = int(self.config.web_control.port)
        try:
            self.web_control.start(host, port)
        except Exception as exc:
            self.web_control_btn.setText("开启 Web 控制")
            self._set_button_variant(self.web_control_btn, "compact")
            self._append_log(f"Web 控制启动失败: {exc}")
            QtWidgets.QMessageBox.critical(self, "Web 控制启动失败", str(exc))
            return False
        self.web_control_btn.setText("关闭 Web 控制")
        self._set_button_variant(self.web_control_btn, "primary")
        display_host = "<本机局域网 IP>" if host in {"0.0.0.0", "::"} else host
        url = f"http://{display_host}:{self.web_control.port}"
        self.web_control_btn.setToolTip(url)
        self._append_log(f"Web 控制已开启: {url}")
        if self._last_preview_bgr is not None:
            self.web_control.update_frame(self._last_preview_bgr)
        return True

    def stop_web_control(self) -> None:
        was_running = self.web_control.is_running
        self.web_control.stop()
        self.web_control_btn.setText("开启 Web 控制")
        self.web_control_btn.setToolTip("")
        self._set_button_variant(self.web_control_btn, "compact")
        if was_running:
            self._append_log("Web 控制已关闭")

    def _web_result(self, ok: bool, message: str, **extra: object) -> dict[str, object]:
        result: dict[str, object] = {"ok": bool(ok), "message": message, "state": self._build_web_state()}
        result.update(extra)
        return result

    def _handle_web_action(self, action: str, payload: dict[str, object]) -> dict[str, object]:
        if action == "state":
            return self._build_web_state()
        if action == "select":
            return self._select_web_target(payload)
        if action == "selection_clear":
            self._clear_web_target()
            return self._web_result(True, "已清除手动选球")
        if action == "capture_start":
            if self.pipeline is None:
                self.start_pipeline()
            return self._web_result(self.pipeline is not None, "采集已启动" if self.pipeline is not None else "采集启动失败")
        if action == "camera_init":
            if self.pipeline is None:
                self.start_pipeline()
            return self._web_result(self.pipeline is not None, "工业相机流程已启动" if self.pipeline is not None else "工业相机流程启动失败")
        if action == "pipeline_init":
            self.initialize_graphics_image_module()
            return self._web_result(True, "图形图像模块初始化命令已执行")
        if action == "projection_start":
            if self.projection_window is None:
                self.toggle_projection_window()
            return self._web_result(self.projection_window is not None, "投影已启动" if self.projection_window is not None else "投影启动失败")
        if action == "camera_refresh":
            self.probe_camera_devices()
            return self._web_result(True, "摄像头列表已刷新")
        if action == "camera_select_default":
            return self._web_result(False, "2606 使用设置中的相机设备，请在设置中修改")
        if action == "match_start":
            if self.pipeline is None:
                return self._web_result(False, "请先开始采集")
            self.reset_state_machine()
            return self._web_result(True, "已重置并开始新局")
        if action == "match_switch_turn":
            switched = self._toggle_turn_target_group(source="web")
            return self._web_result(switched, "已切换到另一花色" if switched else "当前花色尚未确定，无法切换")
        if action in {"match_stage", "match_undo"}:
            return self._web_result(False, "2606 状态机不支持此旧版阶段命令")
        if action == "shot_mode_toggle":
            next_mode = "free" if normalize_shot_mode(self.config.planner.shot_mode) == "rule" else "rule"
            self._set_base_shot_mode(next_mode, source="web")
            return self._web_result(True, "击球模式已切换")
        if action == "shot_mode_set":
            mode = str(payload.get("mode", "")).strip().lower()
            if mode not in {"rule", "free", "free_shot"}:
                return self._web_result(False, f"未知击球模式: {mode}")
            self._set_base_shot_mode(mode, source="web")
            return self._web_result(True, "击球模式已设置")
        if action in {"shot_once_free_arm", "shot_once_free_clear", "shot_once_free_toggle"}:
            enabled = action.endswith("_arm") or (action.endswith("_toggle") and not self.control_state.free_shot_active)
            if enabled:
                self._arm_free_shot_once(source="web")
            else:
                self.control_state.free_shot_active = False
                self._refresh_current_plan()
            return self._web_result(True, "单杆自由击球已开启" if enabled else "单杆自由击球已关闭")
        if action in {"shot_once_black_arm", "shot_once_black_clear", "shot_once_black_toggle"}:
            enabled = action.endswith("_arm") or (action.endswith("_toggle") and not self.control_state.black_shot_active)
            if enabled:
                self._arm_black_shot_once(source="web")
            else:
                self.control_state.black_shot_active = False
                self._refresh_current_plan()
            return self._web_result(self.control_state.black_shot_active == enabled, "单杆黑球已开启" if enabled else "单杆黑球已关闭")
        if action in {"star_formula_toggle", "star_formula_set"}:
            enabled = not self.star_formula.enabled if action.endswith("toggle") else bool(payload.get("enabled", False))
            self._set_star_formula_enabled(enabled, source="web")
            return self._web_result(True, "颗星公式已开启" if enabled else "颗星公式已关闭")
        if action == "instant_replay_export":
            accepted = self._trigger_instant_replay_export(source="web")
            return self._web_result(bool(accepted), "精彩瞬间录制已触发" if accepted else "精彩瞬间录制触发失败")
        if action == "compute":
            self._refresh_current_plan()
            return self._web_result(True, "当前版本实时自动计算路线")
        if action == "detect":
            return self._web_result(True, "当前版本实时自动检测中")
        return self._web_result(False, f"未知动作: {action}")

    def _select_web_target(self, payload: dict[str, object]) -> dict[str, object]:
        if self.last_output is None or self.pipeline is None:
            return self._web_result(False, "当前没有可选轨迹")
        try:
            click = np.asarray([float(payload.get("x", -1)), float(payload.get("y", -1))], dtype=np.float32)
        except (TypeError, ValueError):
            return self._web_result(False, "缺少有效坐标")
        if float(click[0]) < 0.0 or float(click[1]) < 0.0:
            return self._web_result(False, "缺少有效坐标")
        selected = None
        selected_distance = float("inf")
        for track in self.last_output.tracks.tracks:
            if track.group not in {"solid", "stripe", "black"} or track.visibility != "visible":
                continue
            distance = float(np.linalg.norm(np.asarray(track.center_px, dtype=np.float32) - click))
            if distance <= max(18.0, float(track.radius_px) * 2.0) and distance < selected_distance:
                selected = track
                selected_distance = distance
        if selected is None:
            return self._web_result(False, "未命中有效目标球（不可选白球）")
        self._manual_web_target_id = int(selected.track_id)
        self.pipeline.planner.set_manual_target(self._manual_web_target_id)
        self._refresh_current_plan()
        self._append_log(f"Web 手动选中目标球 #{self._manual_web_target_id}")
        return self._web_result(True, f"选中目标球 ID={self._manual_web_target_id}", target_id=self._manual_web_target_id, group=selected.group)

    def _clear_web_target(self) -> None:
        self._manual_web_target_id = None
        if self.pipeline is not None:
            self.pipeline.planner.clear_manual_target()
            self._refresh_current_plan()
        self._append_log("Web 手动选球已清除")

    def _build_web_state(self) -> dict[str, object]:
        out = self.last_output
        frame_w = 0
        frame_h = 0
        tracks: list[dict[str, object]] = []
        if out is not None:
            if out.frame.image is not None:
                frame_h, frame_w = out.frame.image.shape[:2]
            for track in out.tracks.tracks:
                tracks.append(
                    {
                        "track_id": int(track.track_id),
                        "group": str(track.group),
                        "bbox": [float(value) for value in track.bbox],
                        "center": [float(value) for value in track.center_px],
                    }
                )
        rule_route = None
        free_route = None
        base_route_type = normalize_shot_mode(self.config.planner.shot_mode)
        route_type = self.control_state.effective_shot_mode(base_route_type)
        if out is not None and out.plan.best is not None:
            rule_route = to_jsonable(out.plan.best)
            rule_route["pocket_id"] = rule_route.get("pocket_index")
            rule_route["shot_type"] = "rule"
        if out is not None and out.plan.free_route is not None:
            free_route = to_jsonable(out.plan.free_route)
            free_route["pocket_id"] = free_route.get("pocket_index")
            free_route["shot_type"] = "free"
            free_route["score"] = 0.0
        route = free_route if route_type == "free" else rule_route
        layout = out.state.layout if out is not None else []
        visible_solid = sum(1 for track in layout if track.visibility == "visible" and track.group == "solid")
        visible_stripe = sum(1 for track in layout if track.visibility == "visible" and track.group == "stripe")
        last_event = out.state.events[-1].name if out is not None and out.state.events else None
        phase = out.state.phase if out is not None else "IDLE"
        turn_group = self._current_turn_target_group()
        auto_target_id = out.plan.locked_target_id if out is not None else None
        if auto_target_id is None and out is not None and out.plan.best is not None:
            auto_target_id = out.plan.best.target_track_id
        return {
            "ok": True,
            "frame_idx": int(out.frame.frame_id) if out is not None else 0,
            "frame_size": {"w": int(frame_w), "h": int(frame_h)},
            "tracks": tracks,
            "shot_mode": {
                "code": route_type,
                "name": "自由模式" if route_type == "free" else "规则模式",
                "effective_route_type": route_type,
                "base_code": base_route_type,
                "base_name": "自由模式" if base_route_type == "free" else "规则模式",
            },
            "base_shot_mode": {
                "code": base_route_type,
                "name": "自由模式" if base_route_type == "free" else "规则模式",
            },
            "route_type": route_type,
            "route": route,
            "rule_route": rule_route,
            "free_route": free_route,
            "free_status": out.plan.free_status if out is not None else "idle",
            "manual_target_id": self._manual_web_target_id,
            "selected_track_id": self._manual_web_target_id,
            "auto_target_id": auto_target_id,
            "match": {
                "stage_name": phase,
                "turn_group": turn_group,
                "visible_solid": visible_solid,
                "visible_stripe": visible_stripe,
                "remaining_solid": visible_solid,
                "remaining_stripe": visible_stripe,
                "ball_in_hand_notice": False,
                "last_event": last_event,
            },
            "shot_overrides": {
                "free_shot_once": {"active": self.control_state.free_shot_active, "effective": self.control_state.free_shot_active},
                "black_target_once": {"active": self.control_state.black_shot_active, "effective": self.control_state.black_shot_active},
            },
            "pipeline_ready": self.pipeline is not None,
            "capture_running": self.pipeline is not None,
            "projection": {"active": self.projection_window is not None},
            "star_formula_enabled": bool(self.star_formula.enabled),
            "instant_replay_enabled": bool(self.config.instant_replay.enabled),
        }
