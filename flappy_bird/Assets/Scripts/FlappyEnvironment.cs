using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using TMPro;

public class FlappyEnvironment : MonoBehaviour
{
    [SerializeField] private BirdController _bird;
    [SerializeField] private PipeSpawnHandler _pipeSpawner;
    [SerializeField] private GameObject _menu;
    [SerializeField] private GroundObject _ground;
    [SerializeField] private UserInterfaceHandler _ui;
    [SerializeField] private ScoreDisplayScript _scoreDisplay;
    [SerializeField] private TextMeshProUGUI _episodeDisplay;
    private int _episodeCount = 0;

    private TcpClient _client;
    private NetworkStream _stream;
    private bool _connected = false;
    private bool _wasAlive = false;
    private bool _isResetting = false;
    private bool _shouldStartEpisode = false;
    private string _recvBuffer = "";
    private bool _pipePassed = false;
    private Thread _connectionThread;

    void Start()
    {
        _menu.SetActive(false);
    
        _connectionThread = new Thread(() =>
        {
            while (!_connected)
            {
                try
                {
                    _client = new TcpClient("127.0.0.1", 9999);
                    // set read timeout so it never blocks forever
                    _client.ReceiveTimeout = 100;
                    _stream = _client.GetStream();
                    _connected = true;
                    _shouldStartEpisode = true;
                    Debug.Log("Connected to Python!");
                }
                catch
                {
                    Thread.Sleep(500);
                }
            }
        });
        _connectionThread.IsBackground = true;
        _connectionThread.Start();
    }

    void FixedUpdate()
    {
        if (_shouldStartEpisode)
        {
            _shouldStartEpisode = false;
            StartEpisode();
            return;
        }

        if (!_connected || _isResetting) return;

        bool isAlive = _bird.IsAlive;

        if (!isAlive && _isResetting) return;

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

        // latch pipe passed from previous frame
        if (_bird.PassedPipe)
        {
            _pipePassed = true;
            _bird.ResetPassedPipe();
        }

        // read action ONCE
        int action = ReadAction();

        // build reward
        float reward;
        if (!isAlive)
        {
            reward = -1.0f;
        }
        else if (_pipePassed)
        {
            reward = 5.0f;
            _pipePassed = false;
        }
        else
        {
            reward = 0.1f;

            if (action == 1) reward -= 0.1f;

            if (nextPipe != null)
            {
                float distFromGap = Mathf.Abs(birdY - gapY);
                if (distFromGap > 0.5f)
                    reward -= 0.1f;
                else
                    reward += 0.2f;
            }
        }

        int done = isAlive ? 0 : 1;

        // execute action
        if (action == 1 && isAlive)
            _bird.Flap();

        try
        {
            float normBirdY = birdY / 5f;
            float normVelY  = velY / 10f;
            float normPipeX = pipeX / 8f;
            float normGapY  = gapY / 5f;

            string msg = $"{normBirdY},{normVelY},{normPipeX},{normGapY},{reward},{done}\n";
            byte[] data = Encoding.UTF8.GetBytes(msg);
            _stream.Write(data, 0, data.Length);
        }
        catch (System.Exception e)
        {
            Debug.Log($"Socket error: {e.Message}");
        }

        if (!isAlive && _wasAlive && !_isResetting)
        {
            _wasAlive = false;
            _isResetting = true;
            Invoke(nameof(StartEpisode), 0.3f);
        }
    }

    int ReadAction()
    {
        while (true)
        {
            // check if we already have a complete line in buffer
            int newline = _recvBuffer.IndexOf('\n');
            if (newline >= 0)
            {
                string line = _recvBuffer.Substring(0, newline).Trim();
                _recvBuffer = _recvBuffer.Substring(newline + 1);
                return int.Parse(line);
            }

            // read more data
            byte[] buf = new byte[16];
            try
            {
                int bytes = _stream.Read(buf, 0, buf.Length);
                if (bytes > 0)
                    _recvBuffer += Encoding.UTF8.GetString(buf, 0, bytes);
            }
            catch (System.IO.IOException)
            {
                // timeout - return no action
                return 0;
            }
        }
    }

    void StartEpisode()
    {
        _isResetting = false;
        _episodeCount++;
        if (_episodeDisplay != null)
            _episodeDisplay.text = $"Episode: {_episodeCount}";
        _bird.transform.position = new Vector3(-2, 0, 0);
        _bird.rb.linearVelocity = Vector2.zero;
        _pipeSpawner.destroyAllPipes();
        _pipeSpawner.ResetSpawner();
        _scoreDisplay.ResetScore();
        _ground.setScrolling(true);
        _bird.gameObject.SetActive(true);
        _menu.SetActive(false);
        _wasAlive = true;
    }

    void OnApplicationQuit()
    {
        _connected = false;
        _connectionThread?.Abort();
        _stream?.Close();
        _client?.Close();
    }
}