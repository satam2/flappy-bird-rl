using UnityEngine;

public class BirdController : MonoBehaviour
{
    [SerializeField]
    private float _flapForce = 6f;
    public Rigidbody2D rb;
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

    private float _flapCooldown = 0f;
    private float _flapCooldownTime = 0.15f; // minimum time between flaps
    public bool PassedPipe { get; private set; } = false;

    public bool IsAlive
    {
        get { return gameObject.activeSelf; }
    }

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
        gameObject.SetActive(false);
    }

    void Update()
    {
        if(Input.GetMouseButtonDown(0)){
            Flap();
        }
        transform.rotation = Quaternion.Euler(0f, 0f, rb.linearVelocity.y * _rotationSpeed);

        // count down cooldown
        if (_flapCooldown > 0f)
            _flapCooldown -= Time.deltaTime;

        // cap upward velocity
        if (rb.linearVelocity.y > 10f)
            rb.linearVelocity = new Vector2(rb.linearVelocity.x, 10f);
    }

    public void Flap()
    {
        // only flap if cooldown is done
        if (_flapCooldown > 0f) return;
        rb.linearVelocityY = _flapForce;
        _flapCooldown = _flapCooldownTime;
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
        PassedPipe = true;
    }

    public void ResetPassedPipe() 
    {
        PassedPipe = false;
    }
}