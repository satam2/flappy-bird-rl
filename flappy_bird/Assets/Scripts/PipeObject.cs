using UnityEngine;

public class PipeObject: MonoBehaviour
{
    [SerializeField]
    private float moveSpeed = 4f;
    public float deadZone = -5;
    public bool AgentControlled { get; set; }

    void Update()
    {
        if (AgentControlled) return;
        AdvanceSimulation(Time.deltaTime);
    }

    public void AdvanceSimulation(float dt)
    {
        transform.position += Vector3.left * moveSpeed * dt;
        
        if(transform.position.x <= deadZone)
        {
            gameObject.SetActive(false);
            Destroy(gameObject);
        }
    }
}
