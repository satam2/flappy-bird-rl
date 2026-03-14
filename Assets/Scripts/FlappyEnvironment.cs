using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class FlappyEnvironment : MonoBehaviour
{
    [SerializeField] private BirdController _bird;
    [SerializeField] private PipeSpawnHandler _pipeSpawner;
    [SerializeField] private GameObject _menu;
    [SerializeField] private GroundObject _ground;
    [SerializeField] private UserInterfaceHandler _ui;      
    [SerializeField] private ScoreDisplayScript _scoreDisplay; 

    private TcpListener _server;
    private TcpClient _client;
    private NetworkStream _stream;
    private bool _connected = false;
    private bool _wasAlive = false;

    void Start()
    {
        _server = new TcpListener(IPAddress.Loopback, 9999);
        _server.Start();
        Debug.Log("Waiting for Python to connect...");
        _client = _server.AcceptTcpClient();
        _stream = _client.GetStream();
        _connected = true;
        Debug.Log("Python connected!");

        StartEpisode();
    }

    void StartEpisode()
    {
        // Reset bird position and physics
        _bird.transform.position = new Vector3(-2, 0, 0);
        _bird.rb.linearVelocity = Vector2.zero;

        // Reset pipes and score
        _pipeSpawner.destroyAllPipes();
        _scoreDisplay.ResetScore();

        // Use existing UI logic to start — no click needed
        _ui.OnStartPressed();

        _wasAlive = true;
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
            Invoke(nameof(StartEpisode), 0.5f);
        }
    }

    void OnApplicationQuit()
    {
        _stream?.Close();
        _client?.Close();
        _server?.Stop();
    }
}