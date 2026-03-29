from dqn.evaluation import summarize_scores


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
