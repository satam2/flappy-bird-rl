using UnityEngine;
using TMPro;

public class UserInterfaceHandler : MonoBehaviour
{
    [SerializeField]
    private GameObject _bird;
    [SerializeField]
    private GroundObject _ground;
    [SerializeField]
    private Vector3 _playerStartPos = Vector3.zero;
    [SerializeField]
    private GameObject _menuUI;
    [SerializeField]
    private ScoreDisplayScript _scoreDisplay;
    [SerializeField]
    private GameObject _gameOverImg;
    private bool _gameOver = false;

    private void Start()
    {
        _gameOverImg.SetActive(false);
    }

    private void Update()
    {
        if (_gameOver)
        {
            _gameOverImg.SetActive(true);
        }
    }

    public void OnStartPressed()
    {
        _bird.SetActive(true);
        _ground.setScrolling(true);
        _bird.transform.position = _playerStartPos;
        _scoreDisplay.ResetScore();
        _menuUI.SetActive(false);
        _gameOver = true;
    }
}

