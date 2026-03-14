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

    private void Start()
    {
    }

    private void Update()
    {
    }

    public void OnStartPressed()
    {
        _bird.SetActive(true);
        _ground.setScrolling(true);
        _bird.transform.position = _playerStartPos;
        _scoreDisplay.ResetScore();
        _menuUI.SetActive(false);
    }
}

