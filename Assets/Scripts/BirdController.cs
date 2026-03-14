using UnityEngine;

public class BirdController : MonoBehaviour
{
    [SerializeField]
    private float _flapForce = 12f;
    public Rigidbody2D rb;  // NEW: make public for bridge access
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
        get { return gameObject.activeSelf; }
    }

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();  // CHANGED: _rigidBody -> rb
        gameObject.SetActive(false);
    }

    void Update()
    {
        // keep mouse input for manual testing
        if(Input.GetMouseButtonDown(0)){
            Flap();
        }
        transform.rotation = Quaternion.Euler(0f, 0f, rb.linearVelocity.y * _rotationSpeed);
    }

    // callable flap() by the bridge
    public void Flap()
    {
        rb.linearVelocityY = _flapForce;
    }

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
