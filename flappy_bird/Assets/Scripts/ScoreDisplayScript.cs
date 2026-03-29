using UnityEngine;
using UnityEngine.UI;

public class ScoreDisplayScript : MonoBehaviour
{
    [SerializeField]
    private Sprite[] _numSprites;
    [SerializeField]
    private GameObject _digitPrefab;
    [SerializeField]
    private Transform _scoreDisplay;
    [SerializeField]
    private Transform _bestScoreDisplay;

    private int _score = 0;
    private int _bestScore = 0;

    public int CurrentScore => _score;

    void Start()
    {
        UpdateScoreDisplay();
        UpdateBestScoreDisplay();
    }

    public void AddScore()
    {
        _score++;
        UpdateScoreDisplay();

        if (_score > _bestScore)
        {
            _bestScore = _score;
            UpdateBestScoreDisplay();
        }
    }

    public void UpdateScoreDisplay()
    {
        foreach (Transform child in _scoreDisplay)
        {
            Destroy(child.gameObject);
        }

        string scoreStr = _score.ToString();
        foreach (char digitChar in scoreStr)
        {
            int digit = int.Parse(digitChar.ToString());
            GameObject newDigit = Instantiate(_digitPrefab, _scoreDisplay);
            newDigit.GetComponent<Image>().sprite = _numSprites[digit];
        }
    }

    public void UpdateBestScoreDisplay()
    {
        foreach (Transform child in _bestScoreDisplay)
        {
            Destroy(child.gameObject);
        }

        string bestStr = _bestScore.ToString();
        foreach (char digitChar in bestStr)
        {
            int digit = int.Parse(digitChar.ToString());
            GameObject newDigit = Instantiate(_digitPrefab, _bestScoreDisplay);
            newDigit.GetComponent<Image>().sprite = _numSprites[digit];
        }
    }

    public void ResetScore()
    {
        _score = 0;
        UpdateScoreDisplay();
        // best score intentionally not reset
    }
}
