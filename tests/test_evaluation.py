from dqn.evaluation import (
    load_best_eval_mean,
    save_best_eval_mean,
    should_save_best,
    summarize_scores,
)


def test_summarize_scores_reports_mean_median_and_std():
    metrics = summarize_scores([100, 150, 200, 150], early_deaths=1)

    assert metrics["mean_pipes"] == 150.0
    assert metrics["median_pipes"] == 150.0
    assert round(metrics["std_pipes"], 3) == 35.355
    assert metrics["early_deaths"] == 1


def test_summarize_scores_treats_scores_as_pipe_counts():
    metrics = summarize_scores([0, 1, 2, 3], early_deaths=2)
    assert metrics["mean_pipes"] == 1.5
    assert metrics["early_deaths"] == 2


def test_should_save_best_uses_eval_mean_not_training_reward():
    assert should_save_best(current_best=149.0, candidate_mean=150.5) is True
    assert should_save_best(current_best=151.0, candidate_mean=150.5) is False


def test_best_eval_mean_roundtrip_persists_for_resume(tmp_path):
    metric_path = tmp_path / "dqn_best_eval_mean50.txt"

    assert load_best_eval_mean(metric_path) == float("-inf")

    save_best_eval_mean(metric_path, 152.75)

    assert load_best_eval_mean(metric_path) == 152.75


def test_load_best_eval_mean_treats_non_finite_values_as_invalid(tmp_path):
    metric_path = tmp_path / "dqn_best_eval_mean50.txt"
    metric_path.write_text("nan", encoding="utf-8")

    assert load_best_eval_mean(metric_path) == float("-inf")
