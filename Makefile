.PHONY: lock lint test test-contracts test-r1pro test-franka test-plugin test-profile test-profile-libero build

LIBERO_REVISION = 8f1084e3132a39270c3a13ebe37270a43ece2a01

lock:
	uv lock

lint:
	uv run --group test ruff check packages tests integration-tests
	uv run --group test ruff format --check packages tests integration-tests

# 开发机可能安装 ROS 的全局 pytest 插件；这些插件既不属于 Robot SDK，
# 也可能引用另一套 Python 环境。测试入口显式禁止自动加载，保证本地与 CI 一致。
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras python -m pytest -q

test-contracts:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras \
		python -m pytest -q tests/conformance/test_canonical_fixtures.py

# 真实模型测试不把大型资产复制进 SDK 仓。CI 和开发者显式指定干净资产根目录，
# 缺少资产时立即失败，不能用 pytest skip 冒充已验证。
test-r1pro:
	test -n "$$R1PRO_ASSET_ROOT"
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras \
		python -m pytest -q integration-tests/test_r1pro_model.py

# Franka 大型模型不进入 SDK 或 Python 环境。该门控要求经过许可证与 manifest
# 审查的独立资产包，并真实运行 Pinocchio FK/IK 和 Ruckig，不能用 skip 代替。
test-franka:
	test -n "$$FRANKA_MODEL_ROOT"
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras \
		python -m pytest -q integration-tests/test_franka_model.py

# 需要调用方先启动真实 Plugin Runtime。测试会自行创建并停止场景实例。
test-plugin:
	test -n "$$R1PRO_ASSET_ROOT"
	test -n "$$PLUGIN_MUJOCO_URL"
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras \
		python -m pytest -q integration-tests/test_plugin_runtime.py

# 连接真实 robosuite 或 LIBERO Profile Runtime，验证 Franka 的公共 Backend 行为。
test-profile:
	test -n "$$PLUGIN_MUJOCO_PROFILE_URL"
	test -n "$$FRANKA_MODEL_ROOT"
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras \
		python -m pytest -q integration-tests/test_profile_runtime.py

# 固定 LIBERO Profile 的 Franka Backend 行为；场景、layout 与 profile ID
# 不能由 Runner 默认值漂移，失败时保留 JUnit 证据。
test-profile-libero:
	test -n "$$PLUGIN_MUJOCO_PROFILE_URL"
	test -n "$$FRANKA_MODEL_ROOT"
	test -n "$$SEMANTIC_LIBERO_ROOT"
	test -n "$$SEMANTIC_LIBERO_REVISION"
	test "$$SEMANTIC_LIBERO_REVISION" = "$(LIBERO_REVISION)"
	test "$$(git -C "$$SEMANTIC_LIBERO_ROOT" rev-parse HEAD)" = "$$SEMANTIC_LIBERO_REVISION"
	test -n "$$SIMULATION_EVIDENCE_DIR"
	mkdir -p "$$SIMULATION_EVIDENCE_DIR/sdk"
	PLUGIN_MUJOCO_PROFILE_ID=libero-robosuite-1.4 \
		PLUGIN_MUJOCO_PROFILE_SCENE=libero_spatial:0 PLUGIN_MUJOCO_PROFILE_LAYOUT=init-0 \
		PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group test --all-extras python -m pytest -q \
		--junitxml="$$SIMULATION_EVIDENCE_DIR/sdk/libero-profile.xml" integration-tests/test_profile_runtime.py

build:
	uv build --all-packages
