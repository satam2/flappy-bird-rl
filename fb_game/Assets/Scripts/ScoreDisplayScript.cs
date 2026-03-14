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
    private int _score = 0;

    void Start()
    {
        UpdateScoreDisplay();
    }
    public void AddScore()
    {
        _score++;
        UpdateScoreDisplay();
    }

    public void UpdateScoreDisplay()
    {
        foreach (Transform chlid in _scoreDisplay)
        {
            Destroy(chlid.gameObject);
        }

        string scoreStr = _score.ToString();
        foreach (char digitChar in scoreStr)
        {
            int digit = int.Parse(digitChar.ToString());
            GameObject newDigit = Instantiate(_digitPrefab, _scoreDisplay);
            newDigit.GetComponent<Image>().sprite = _numSprites[digit];
        }
    }

    public void ResetScore()
    {
        _score = 0;
        UpdateScoreDisplay();
    }
}
