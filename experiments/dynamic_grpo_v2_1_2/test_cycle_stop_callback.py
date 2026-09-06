from __future__ import annotations

from types import SimpleNamespace

from adaptive_trainer import AdaptiveTrainerConfig, make_adaptive_grpo_trainer


def test_cycle_boundary_stops_without_shortening_scheduler_horizon() -> None:
    class BaseTrainer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.args = SimpleNamespace(
                num_generations=8,
                generation_batch_size=8,
                num_iterations=2,
                scale_rewards="none",
                loss_type="dapo",
                epsilon=0.2,
                epsilon_high=0.2,
                delta=None,
                max_steps=149,
            )
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    Trainer = make_adaptive_grpo_trainer(
        BaseTrainer,
        AdaptiveTrainerConfig(num_iterations=2, cycle_stop_global_step=16),
    )
    trainer = Trainer()
    assert trainer.args.max_steps == 149
    assert len(trainer.callbacks) == 1
    control = SimpleNamespace(should_training_stop=False)
    trainer.callbacks[0].on_step_end(
        trainer.args,
        SimpleNamespace(global_step=15),
        control,
    )
    assert control.should_training_stop is False
    trainer.callbacks[0].on_step_end(
        trainer.args,
        SimpleNamespace(global_step=16),
        control,
    )
    assert control.should_training_stop is True

