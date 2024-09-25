using UnityEngine;

public class MoveBall : MonoBehaviour
{
    public float speed = 10f; // Speed of the ball

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>(); // Get the Rigidbody attached to the ball
    }

    void FixedUpdate()
    {
        // Get input from the player (keyboard or controller)
        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");

        // Create a vector for movement
        Vector3 movement = new Vector3(moveHorizontal, 0.0f, moveVertical);

        // Apply the movement to the ball using physics
        rb.AddForce(movement * speed);
    }
}
