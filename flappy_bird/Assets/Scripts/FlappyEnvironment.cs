using System;
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
    [SerializeField] private string _host = "127.0.0.1";
    [SerializeField] private int _port = 9999;

    [Serializable] private class Request
    {
        public string command;
        public int action;
        public int seed;
    }

    [Serializable] private class Response
    {
        public int protocol = 2;
        public float[] state;
        public float reward;
        public bool done;
        public int score;
        public string error;
    }

    private TcpClient _client;
    private NetworkStream _stream;
    private Thread _connectionThread;
    private volatile bool _running;
    private volatile bool _connected;
    private bool _hasEpisode;
    private int _episodeCount;
    private string _recvBuffer = "";
    private SimulationMode2D _previousSimulationMode;
    private const float StepSeconds = 0.02f;

    private void Start()
    {
        _menu.SetActive(false);
        _bird.AgentControlled = true;
        _pipeSpawner.AgentControlled = true;
        _previousSimulationMode = Physics2D.simulationMode;
        Physics2D.simulationMode = SimulationMode2D.Script;
        _running = true;
        _connectionThread = new Thread(Connect) { IsBackground = true };
        _connectionThread.Start();
    }

    private void Connect()
    {
        while (_running && !_connected)
        {
            var candidate = new TcpClient();
            try
            {
                candidate.Connect(_host, _port);
                candidate.NoDelay = true;
                if (!_running) { candidate.Close(); return; }
                _client = candidate;
                _stream = candidate.GetStream();
                _connected = true;
            }
            catch (SocketException)
            {
                candidate.Close();
                Thread.Sleep(200);
            }
        }
    }

    private void Update()
    {
        if (!_connected) return;
        try
        {
            if (_client.Client.Poll(0, SelectMode.SelectRead) && _client.Available == 0)
                throw new System.IO.IOException("Python disconnected");
            while (_stream.DataAvailable)
            {
                byte[] bytes = new byte[4096];
                int count = _stream.Read(bytes, 0, bytes.Length);
                if (count == 0) throw new System.IO.IOException("Python disconnected");
                _recvBuffer += Encoding.UTF8.GetString(bytes, 0, count);
                if (_recvBuffer.Length > 65536)
                    throw new System.IO.IOException("Request exceeds protocol limit");
            }
            int newline;
            while ((newline = _recvBuffer.IndexOf('\n')) >= 0)
            {
                string line = _recvBuffer.Substring(0, newline);
                _recvBuffer = _recvBuffer.Substring(newline + 1);
                Response response;
                try { response = Handle(JsonUtility.FromJson<Request>(line)); }
                catch (Exception e) { response = new Response { error = e.Message }; }
                byte[] data = Encoding.UTF8.GetBytes(JsonUtility.ToJson(response) + "\n");
                _stream.Write(data, 0, data.Length);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"RL connection stopped: {e.Message}. Restart Play mode to reconnect.");
            _connected = false;
            _stream?.Close();
            _client?.Close();
        }
    }

    private Response Handle(Request request)
    {
        if (request == null) throw new ArgumentException("Missing request");
        if (request.command == "reset")
        {
            UnityEngine.Random.InitState(request.seed);
            _pipeSpawner.destroyAllPipes();
            _bird.ResetEpisode(new Vector3(-2, 0, 0));
            _scoreDisplay.ResetScore();
            _pipeSpawner.ResetSpawner();
            _pipeSpawner.SpawnInitialPipe();
            _ground.setScrolling(true);
            _menu.SetActive(false);
            _hasEpisode = true;
            _episodeCount++;
            if (_episodeDisplay != null) _episodeDisplay.text = $"Episode: {_episodeCount}";
            Physics2D.SyncTransforms();
            return Observe(0f);
        }
        if (request.command != "step" || (request.action != 0 && request.action != 1))
            throw new ArgumentException("Expected reset or step with action 0/1");
        if (!_hasEpisode || !_bird.IsAlive)
            throw new InvalidOperationException("Episode finished; reset before step");

        // No simulation runs while Python is thinking. One request advances one tick.
        _bird.ResetPassedPipe();
        _bird.AdvanceSimulation(request.action, StepSeconds);
        _pipeSpawner.MovePipes(StepSeconds);
        Physics2D.SyncTransforms();
        Physics2D.Simulate(StepSeconds);
        _bird.FinishSimulationStep(StepSeconds);
        _bird.CommitPipePass();
        _pipeSpawner.FinishSimulationStep(StepSeconds);

        float reward;
        if (!_bird.IsAlive) reward = -5f;
        else if (_bird.PassedPipe) reward = 10f;
        else
        {
            reward = 0.1f - 0.1f * request.action;
            GameObject pipe = _pipeSpawner.GetNextPipe(_bird.transform.position.x);
            if (pipe != null)
                reward += Mathf.Abs(_bird.transform.position.y - _pipeSpawner.GetGapCenterY(pipe)) <= 0.8f
                    ? 0.2f : -0.1f;
        }
        return Observe(reward);
    }

    private Response Observe(float reward)
    {
        GameObject pipe = _pipeSpawner.GetNextPipe(_bird.transform.position.x);
        return new Response
        {
            state = new float[]
            {
                _bird.transform.position.y / 5f, _bird.rb.linearVelocity.y / 10f,
                pipe != null ? (pipe.transform.position.x - _bird.transform.position.x) / 8f : 0f,
                pipe != null ? _pipeSpawner.GetGapCenterY(pipe) / 5f : 0f,
                _bird.NormalizedCooldown
            },
            reward = reward, done = !_bird.IsAlive, score = _scoreDisplay.Score
        };
    }

    private void OnDestroy()
    {
        _running = false;
        _connected = false;
        _stream?.Close();
        _client?.Close();
        _connectionThread?.Join(250);
        Physics2D.simulationMode = _previousSimulationMode;
    }
}
