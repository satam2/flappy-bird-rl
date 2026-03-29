using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class PipeSpawnHandler : MonoBehaviour
{
    [SerializeField]
    private GameObject _pipe;
    private List<GameObject> _spawnedPipes = new List<GameObject>();
    private float _spawnRate = 1f;
    private float _timer = 0f;
    private float _offsetMin = -2, _offsetMax = 2;
    [SerializeField]
    private BirdController _bird;

    void Update()
    {
        _spawnedPipes.RemoveAll(pipe => pipe == null);

        if (_timer <= 0f && _bird.IsAlive){
            spawnPipe();
            _timer = _spawnRate;
        }
        else
        {
            _timer -= Time.deltaTime;
        }
    }

    public void ResetSpawner()
    {
        _timer = 0f;  // spawn immediately on next Update
    }

    void spawnPipe()
    {
        Vector3 newPos = new Vector3(transform.position.x, Random.Range(_offsetMin, _offsetMax));
        GameObject newPipe = Instantiate(_pipe, newPos, transform.rotation);
        _spawnedPipes.Add(newPipe);
    }

    public void destroyAllPipes()
    {
        foreach(GameObject pipe in _spawnedPipes)
        {
            Destroy(pipe);
        }
        _spawnedPipes.Clear();
    }

    // get the next pipe ahead of the bird
    public GameObject GetNextPipe(float birdX)
    {
        TryGetUpcomingPipes(birdX, out GameObject nextPipe, out _);
        return nextPipe;
    }

    public bool TryGetUpcomingPipes(float birdX, out GameObject nextPipe, out GameObject followingPipe)
    {
        _spawnedPipes.RemoveAll(pipe => pipe == null);

        List<GameObject> ordered = _spawnedPipes
            .Where(pipe => pipe != null && pipe.transform.position.x - birdX > -1f)
            .OrderBy(pipe => pipe.transform.position.x)
            .ToList();

        nextPipe = ordered.Count > 0 ? ordered[0] : null;
        followingPipe = ordered.Count > 1 ? ordered[1] : null;
        return nextPipe != null;
    }

    // get the Y center of the gap between TopPipe and BottomPipe
    public float GetGapCenterY(GameObject pipe)
    {
        Transform point = pipe.transform.Find("Point");
        if (point == null)
        {
            return pipe.transform.position.y;
        }
        return point.position.y;
    }
}
