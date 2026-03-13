using UnityEngine;
using System.Collections.Generic;

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
        if (_timer <= 0f && _bird.IsAlive){
            spawnPipe();
            _timer = _spawnRate;
        }
        else
        {
            _timer -= Time.deltaTime;
        }
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
}
