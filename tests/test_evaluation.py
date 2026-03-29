from dqn.evaluation import should_save_best, summarize_scores


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
