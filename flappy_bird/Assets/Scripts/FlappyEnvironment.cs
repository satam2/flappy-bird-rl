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
    private string _recvBuffer = "";
    private bool _pipePassed = false;
    private Thread _connectionThread;
    private float _lastReward = 0f;
    private int _lastDone = 0;

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
        if (!_connected || _stream == null) return;
        if (!TryReadLine(out string line)) return;
        HandleCommand(line);
    }

    private bool TryReadLine(out string line)
    {
        line = "";

        while (true)
        {
            int newline = _recvBuffer.IndexOf('\n');
            if (newline >= 0)
            {
                line = _recvBuffer.Substring(0, newline).Trim();
                _recvBuffer = _recvBuffer.Substring(newline + 1);
                return true;
            }

            byte[] buf = new byte[16];
            try
            {
                int bytes = _stream.Read(buf, 0, buf.Length);
                if (bytes > 0)
                {
                    _recvBuffer += Encoding.UTF8.GetString(buf, 0, bytes);
                    continue;
                }
                return false;
            }
            catch (System.IO.IOException)
            {
                return false;
            }
        }
    }

    private void HandleCommand(string line)
    {
        if (string.IsNullOrEmpty(line)) return;

        string[] parts = line.Split('|');
        if (parts.Length == 0) return;

        switch (parts[0])
        {
            case "RESET":
                string mode = parts.Length > 1 ? parts[1] : "train";
                int seed = 0;
                if (parts.Length > 2)
                {
                    int.TryParse(parts[2], out seed);
                }
                ResetEpisode(mode, seed);
                SendCurrentPacket();
                break;
            case "STEP":
                if (parts.Length < 2) return;
                if (!int.TryParse(parts[1], out int action)) return;
                StepEpisode(action);
                SendCurrentPacket();
                break;
        }
    }

    private void ResetEpisode(string mode, int seed)
    {
        Random.InitState(seed);
        StartEpisode();
        _lastReward = 0f;
        _lastDone = 0;
        _pipePassed = false;
    }

    private void StepEpisode(int action)
    {
        float previousAbsDy = 0f;
        GameObject previousPipe = _pipeSpawner.GetNextPipe(_bird.transform.position.x);
        if (previousPipe != null)
        {
            float previousGapY = _pipeSpawner.GetGapCenterY(previousPipe);
            previousAbsDy = Mathf.Abs(_bird.transform.position.y - previousGapY);
        }

        if (action == 1 && _bird.IsAlive)
        {
            _bird.Flap();
        }

        if (_bird.PassedPipe)
        {
            _pipePassed = true;
            _bird.ResetPassedPipe();
        }

        float currentAbsDy = 0f;
        GameObject currentPipe = _pipeSpawner.GetNextPipe(_bird.transform.position.x);
        if (currentPipe != null)
        {
            float currentGapY = _pipeSpawner.GetGapCenterY(currentPipe);
            currentAbsDy = Mathf.Abs(_bird.transform.position.y - currentGapY);
        }

        float shaping = Mathf.Clamp(previousAbsDy - currentAbsDy, -0.03f, 0.03f);
        float reward = shaping;

        if (_pipePassed)
        {
            reward += 1.0f;
            _pipePassed = false;
        }

        if (!_bird.IsAlive)
        {
            reward = -2.0f;
        }

        _lastReward = reward;
        _lastDone = _bird.IsAlive ? 0 : 1;
    }

    private void SendCurrentPacket()
    {
        if (_stream == null) return;

        float birdX = _bird.transform.position.x;
        float birdY = _bird.transform.position.y;
        float velY = _bird.rb.linearVelocity.y;
        float dxNext = 0f;
        float dyNext = 0f;
        float dxFollowing = 0f;
        float dyFollowing = 0f;

        if (_pipeSpawner.TryGetUpcomingPipes(birdX, out GameObject nextPipe, out GameObject followingPipe))
        {
            dxNext = nextPipe.transform.position.x - birdX;
            dyNext = _pipeSpawner.GetGapCenterY(nextPipe) - birdY;

            if (followingPipe != null)
            {
                dxFollowing = followingPipe.transform.position.x - birdX;
                dyFollowing = _pipeSpawner.GetGapCenterY(followingPipe) - birdY;
            }
        }

        float canFlap = _bird.CanFlap ? 1f : 0f;
        float normBirdY = birdY / 5f;
        float normVelY = velY / 10f;
        float normDxNext = dxNext / 8f;
        float normDyNext = dyNext / 5f;
        float normDxFollowing = dxFollowing / 8f;
        float normDyFollowing = dyFollowing / 5f;

        int pipesPassed = _scoreDisplay != null ? _scoreDisplay.CurrentScore : 0;
        string msg = $"{normBirdY},{normVelY},{normDxNext},{normDyNext},{normDxFollowing},{normDyFollowing},{canFlap},{_lastReward},{_lastDone},{pipesPassed}\n";

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(msg);
            _stream.Write(data, 0, data.Length);
        }
        catch (System.Exception e)
        {
            Debug.Log($"Socket error: {e.Message}");
        }
    }

    void StartEpisode()
    {
        _episodeCount++;
        if (_episodeDisplay != null)
            _episodeDisplay.text = $"Episode: {_episodeCount}";
        _bird.transform.position = new Vector3(-2, 0, 0);
        _bird.rb.linearVelocity = Vector2.zero;
        _bird.ResetFlapCooldown();
        _pipeSpawner.destroyAllPipes();
        _pipeSpawner.ResetSpawner();
        _scoreDisplay.ResetScore();
        _ground.setScrolling(true);
        _bird.gameObject.SetActive(true);
        _menu.SetActive(false);
        _bird.ResetPassedPipe();
    }

    void OnApplicationQuit()
    {
        _connected = false;
        _connectionThread?.Abort();
        _stream?.Close();
        _client?.Close();
    }
}
