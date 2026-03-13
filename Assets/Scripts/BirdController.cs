using UnityEngine;

public class BirdController : MonoBehaviour
{
    [SerializeField]
    private float _flapForce = 12f;
    private Rigidbody2D _rigidBody;
    [SerializeField]
    private GameObject _menu;
    [SerializeField]
    private ScoreDisplayScript _scoreDisplay;
    [SerializeField]
    private float _rotationSpeed = 1.5f;
    [SerializeField]
    private PipeSpawnHandler _pipeSpawner;
    [SerializeField]
    private GroundObject _ground;

    public bool IsAlive
    {
        get
        {
            return gameObject.activeSelf;
        }
    }

    private void Awake()
    {
        _rigidBody = GetComponent<Rigidbody2D>();
        gameObject.SetActive(false);
    }

    void Update()
    {
        if(Input.GetMouseButtonDown(0)){
        _rigidBody.linearVelocityY = _flapForce;
        }
        transform.rotation = Quaternion.Euler(0f, 0f, _rigidBody.linearVelocity.y * _rotationSpeed);
    }

    // obj w rigidbody and collision encounters another obj w collision
    private void OnCollisionEnter2D(Collision2D collision)
    {
        _pipeSpawner.destroyAllPipes();
        gameObject.SetActive(false);
        _ground.setScrolling(false);
        _menu.SetActive(true);
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        _scoreDisplay.AddScore();
    }
}
