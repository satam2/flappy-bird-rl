using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class FlappyEnvironment : MonoBehaviour
{
    [SerializeField] private BirdController _bird;
    [SerializeField] private PipeSpawnHandler _pipeSpawner;
    [SerializeField] private GameObject _menu;
    [SerializeField] private GroundObject _ground;
    [SerializeField] private UserInterfaceHandler _ui;
    [SerializeField] private ScoreDisplayScript _scoreDisplay;

    private TcpClient _client;
    private NetworkStream _stream;
    private bool _connected = false;
    private bool _wasAlive = false;
    private bool _isResetting = false; 
    private Thread _connectionThread;

    void Start()
    {
        Debug.Log("Connecting to Python...");
        _menu.SetActive(false);
        // Connect to Python server on background thread
        _connectionThread = new Thread(() =>
        {
            while (!_connected)
            {
                try
                {
                    _client = new TcpClient("127.0.0.1", 9999);
                    _stream = _client.GetStream();
                    _connected = true;
                    Debug.Log("Connected to Python!");
                }
                catch
                {
                    // Python not ready yet, keep trying
                    Thread.Sleep(500);
                }
            }
        });
        _connectionThread.IsBackground = true;
        _connectionThread.Start();
    }

    void FixedUpdate()
    {
        if (!_connected) return;

        bool isAlive = _bird.IsAlive;

        float birdY = _bird.transform.position.y;
        float velY  = _bird.rb.linearVelocity.y;
        float pipeX = 0f;
        float gapY  = 0f;

        GameObject nextPipe = _pipeSpawner.GetNextPipe(_bird.transform.position.x);
        if (nextPipe != null)
        {
            pipeX = nextPipe.transform.position.x - _bird.transform.position.x;
            gapY  = _pipeSpawner.GetGapCenterY(nextPipe);
        }

        float reward = isAlive ? 0.1f : -1.0f;
        int done     = isAlive ? 0 : 1;

        string msg = $"{birdY},{velY},{pipeX},{gapY},{reward},{done}\n";
        byte[] data = Encoding.UTF8.GetBytes(msg);
        _stream.Write(data, 0, data.Length);

        byte[] buf = new byte[16];
        int bytes  = _stream.Read(buf, 0, buf.Length);
        int action = int.Parse(Encoding.UTF8.GetString(buf, 0, bytes).Trim());

        if (action == 1 && isAlive)
            _bird.Flap();

        if (!isAlive && _wasAlive)
        {
            _wasAlive = false;
            _isResetting = true;
            Invoke(nameof(StartEpisode), 1.5f);
        }
    }

    void StartEpisode(){
        _isResetting = false;
        _bird.transform.position = new Vector3(-2, 0, 0);
        _bird.rb.linearVelocity = Vector2.zero;
        _pipeSpawner.destroyAllPipes();
        _pipeSpawner.ResetSpawner();
        _scoreDisplay.ResetScore();
        _ground.setScrolling(true);       // start ground scrolling
        _bird.gameObject.SetActive(true); // activate bird directly
        _menu.SetActive(false);           // keep menu hidden
        _wasAlive = true;
    }

    void OnApplicationQuit()
    {
        _connectionThread?.Abort();
        _stream?.Close();
        _client?.Close();
    }
}