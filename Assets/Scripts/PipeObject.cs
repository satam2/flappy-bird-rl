using UnityEngine;

public class PipeObject: MonoBehaviour
{
    [SerializeField]
    private float moveSpeed = 4f;
    public float deadZone = -5;

    void Update()
    {
        transform.position += Vector3.left * moveSpeed * Time.deltaTime;
        
        if(transform.position.x <= deadZone)
        {

            Destroy(gameObject);
        }
    }
}
