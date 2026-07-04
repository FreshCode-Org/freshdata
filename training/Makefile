# FreshData development-time training pipeline (Phase 5).
# All targets run from the repository root; `make -C training <target>`
# and the root Makefile's training-* passthrough both work.

PY ?= python
ROOT := ..

.PHONY: help training-seed training-corrupt training-teacher-labels \
        training-distill training-eval training-package-artifacts \
        training-dev-artifacts training-release-artifacts

help:
	@echo "Targets:"
	@echo "  training-seed               Validate licenses + build the seed corpus"
	@echo "  training-corrupt            Generate labeled corrupted training data"
	@echo "  training-teacher-labels     Run compliance-gated teacher tasks (degrades offline)"
	@echo "  training-distill            Train role/intent heads (+ optional encoder), export, quantize"
	@echo "  training-eval               Calibration eval + full CleanBench release gates"
	@echo "  training-package-artifacts  Package artifacts into dist/artifacts/"
	@echo "  training-dev-artifacts      Small synthetic-only end-to-end build (no cloud)"
	@echo "  training-release-artifacts  Full release pipeline; fails on any gate failure"

training-seed:
	cd $(ROOT) && $(PY) -m training.datasets.validators --check-licenses
	cd $(ROOT) && $(PY) -m training.datasets.build_seed

training-corrupt:
	cd $(ROOT) && $(PY) -m training.datasets.build_corrupted

training-teacher-labels:
	cd $(ROOT) && $(PY) -m training.teacher.compliance check
	@echo "teacher tasks run on demand via training.teacher.tasks.run_task;"
	@echo "cached outputs (training/cache/teacher/) feed the trainers automatically."

training-distill:
	cd $(ROOT) && $(PY) -m training.distill.train_role_head --check-gates
	cd $(ROOT) && $(PY) -m training.distill.train_intent_head --check-gates
	cd $(ROOT) && $(PY) -m training.distill.train_encoder_contrastive --check-gates
	cd $(ROOT) && $(PY) -m training.distill.export_onnx
	cd $(ROOT) && $(PY) -m training.distill.quantize --check-gates

training-eval:
	cd $(ROOT) && $(PY) -m training.calibration.build_features
	cd $(ROOT) && $(PY) -m training.calibration.fit_isotonic --fit-deterministic
	cd $(ROOT) && $(PY) -m training.calibration.export_tables
	cd $(ROOT) && $(PY) -m training.calibration.eval_calibration --check-gates
	cd $(ROOT) && $(PY) -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --check-gates

training-package-artifacts:
	cd $(ROOT) && $(PY) -m training.distill.package_artifacts

# Small synthetic-only artifact set: no cloud teacher calls, no onnx needed,
# nothing enters the wheel. Ends with the CleanBench mini gates (T1-T4).
training-dev-artifacts:
	cd $(ROOT) && $(PY) -m training.datasets.validators --check-licenses
	cd $(ROOT) && $(PY) -m training.datasets.build_seed
	cd $(ROOT) && $(PY) -m training.datasets.build_corrupted --dev
	cd $(ROOT) && $(PY) -m training.distill.train_role_head --dev --check-gates
	cd $(ROOT) && $(PY) -m training.distill.train_intent_head --dev --check-gates
	cd $(ROOT) && $(PY) -m training.distill.export_onnx
	cd $(ROOT) && $(PY) -m training.distill.quantize --check-gates
	cd $(ROOT) && $(PY) -m training.calibration.build_features --seeds 3
	cd $(ROOT) && $(PY) -m training.calibration.fit_isotonic --fit-deterministic
	cd $(ROOT) && $(PY) -m training.calibration.export_tables
	cd $(ROOT) && $(PY) -m training.calibration.eval_calibration --check-gates
	cd $(ROOT) && $(PY) -m training.distill.package_artifacts
	cd $(ROOT) && $(PY) -m benchmarks.cleanbench --tracks T1,T2,T3,T4 --check-gates

# Full release pipeline: compliance first, human-verified eval enforced by
# the trainers, ONNX required, every gate blocking.
training-release-artifacts:
	cd $(ROOT) && $(PY) -m training.teacher.compliance check
	cd $(ROOT) && $(PY) -m training.datasets.validators --check-licenses
	cd $(ROOT) && $(PY) -m training.datasets.build_seed
	cd $(ROOT) && $(PY) -m training.datasets.build_corrupted
	cd $(ROOT) && $(PY) -m training.distill.train_role_head --check-gates
	cd $(ROOT) && $(PY) -m training.distill.train_intent_head --check-gates
	cd $(ROOT) && $(PY) -m training.distill.train_encoder_contrastive --check-gates
	cd $(ROOT) && $(PY) -m training.distill.export_onnx --release
	cd $(ROOT) && $(PY) -m training.distill.quantize --check-gates
	cd $(ROOT) && $(PY) -m training.calibration.build_features
	cd $(ROOT) && $(PY) -m training.calibration.fit_isotonic --fit-deterministic
	cd $(ROOT) && $(PY) -m training.calibration.export_tables
	cd $(ROOT) && $(PY) -m training.calibration.eval_calibration --check-gates
	cd $(ROOT) && $(PY) -m training.distill.package_artifacts --release
	cd $(ROOT) && $(PY) -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --check-gates
	@echo "release pipeline complete: dist/artifacts/ + benchmarks/results/latest.*"
