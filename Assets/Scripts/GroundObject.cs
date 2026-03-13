using UnityEngine;

public class GroundObject : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    [SerializeField]
    private float _scrollSpeed = 0.5f;
    private bool _isScrolling = false;
    private Material _material;
    private Vector2 _offset;
    void Start()
    {
        SpriteRenderer renderer = GetComponent<SpriteRenderer>();
        _material = renderer.material;
    }

    // Update is called once per frame
    void Update()
    {
        if(!_isScrolling) return;
        _offset.x += _scrollSpeed * Time.deltaTime;
        _material.mainTextureOffset = _offset;
    }

    public void setScrolling(bool tf)
    {
        _isScrolling = tf;
    }
}
