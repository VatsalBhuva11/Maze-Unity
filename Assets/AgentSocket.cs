using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class AgentSocket : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    public Transform target;
    public Transform start;
    private TcpListener server;
    public float moveSpeed = 1f;

    private Vector3 initialAgentPosition;
    private Vector3 initialTargetPosition;
    private bool episodeDone = false;
    private int episodeCount = 0;
    private int steps = 0;
    private Vector3 origin = new Vector3(0f, 0f, 0f);

    private void Start()
    {
        GameObject targetObject = GameObject.FindGameObjectWithTag("target");
        GameObject startObject = GameObject.FindGameObjectWithTag("start");
        episodeDone = false;
        Time.timeScale = 10f;

        if (targetObject != null) target = targetObject.transform;
        else
        {
            Debug.LogError("Target GameObject with tag 'target' not found.");
            return;
        }

        if (startObject != null) start = startObject.transform;
        else
        {
            Debug.LogError("Start GameObject with tag 'start' not found.");
            return;
        }

        initialAgentPosition = start.position;
        transform.position = initialAgentPosition;
        initialTargetPosition = target.position;

        /*

        float checkRadius = 0.5f; // Adjust radius as needed

        // Get all colliders within the radius at the current position
        Vector3 currentPosition = transform.position;
        currentPosition.y = 0f;
        Collider[] colliders = Physics.OverlapSphere(currentPosition, checkRadius);

        // Iterate through the colliders and log their positions
        foreach (Collider collider in colliders)
        {
            // Log the name and position of the collider
            Debug.Log($"Collider Name: {collider.name}, Position: {collider.transform.position}");
        }

        // If no colliders are found, log a message
        if (colliders.Length == 0)
        {
            Debug.Log("No colliders detected in the specified radius.");
        }

        */

        StartServer();
    }

    private void StartServer()
    {
        try
        {
            server = new TcpListener(System.Net.IPAddress.Parse("127.0.0.1"), 4657);
            server.Start();
            Debug.Log("Waiting for Python to connect...");
            server.BeginAcceptTcpClient(OnClientConnected, null);
        }
        catch (Exception e)
        {
            Debug.Log("Server error: " + e);
        }
    }

    private void OnClientConnected(IAsyncResult ar)
    {
        try
        {
            client = server.EndAcceptTcpClient(ar);
            stream = client.GetStream();
            Debug.Log("Python connected successfully.");
        }
        catch (Exception e)
        {
            Debug.Log("Client connection error: " + e);
        }
    }

    private void OnApplicationQuit()
    {
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }

    
    private void Update()
    {
        if (stream != null && stream.CanRead && stream.CanWrite)
        {

            string action = ReceiveDataFromPython();
            if (!string.IsNullOrEmpty(action))
            {
                Debug.Log("Received action from Python: " + action);
                if (action == "RESET")
                {
                    ResetPositions();
                    string initialPos = CollectObservations();
                    SendDataToPython(initialPos);
                    return;
                }
                
                PerformAction(action);
                
            }

            string observations = CollectObservations();
            SendDataToPython(observations);
            
            steps++;
            if (steps >= 200 || episodeDone)
            {
                episodeCount++;
                Debug.Log($"----- Episode {episodeCount}, Steps: {steps}, Done: {episodeDone} -----");
                ResetPositions();
                steps = 0;
                episodeDone = false;
            }

        }
    }

    

    private string CollectObservations()
    {
        Vector3 agentRelativePosition = transform.localPosition - origin;
        Vector3 targetRelativePosition = target.localPosition - origin;

        float agentX = agentRelativePosition.x;
        float agentY = agentRelativePosition.z;
        float targetX = targetRelativePosition.x;
        float targetY = targetRelativePosition.z;
        float done = episodeDone ? 1 : 0;
        // Debug.Log($"collect obs: {agentX},{agentY},{targetX},{targetY}");
        return $"{agentX},{agentY},{targetX},{targetY},{done}";
    }

    private void PerformAction(string action)
    {
        float moveStep = 1.0f; // Movement step size
        Vector3 move = Vector3.zero;
        Quaternion targetRotation = transform.localRotation;

        switch (action)
        {
            case "0":
                break; // No action
            case "1":
                move = new Vector3(0, 0, moveStep); // right (+ve z)
                targetRotation = Quaternion.Euler(0, 360, 0); 
                break;
            case "2":
                move = new Vector3(moveStep, 0, 0); // down (+ve x)
                targetRotation = Quaternion.Euler(0, 270, 0);
                break;
            case "3":
                move = new Vector3(0, 0, -moveStep); // left (-ve z)
                targetRotation = Quaternion.Euler(0, 180, 0); 
                break;
            case "4":
                move = new Vector3(-moveStep, 0, 0); // up (-ve x)
                targetRotation = Quaternion.Euler(0, 90, 0);
                break;
        }

        // Apply rotation first to face the direction
        transform.localRotation = targetRotation;
        Vector3 newPosition;
        // Apply movement
        if (move != Vector3.zero)
        {
            newPosition = transform.localPosition + move;
            Debug.Log($"Trying to move to: ({newPosition.x},{newPosition.z}");

            if (CheckForWallCollision(transform.localPosition, newPosition))
            {
                episodeDone = true;
                Debug.Log("Agent hit a wall. Ending episode.");
            }
            else
            {
                transform.localPosition = newPosition;
                CheckGoal();
            }
        }
    }
    private bool CheckForWallCollision(Vector3 currentPosition, Vector3 newPosition)
    {
        // Calculate the direction and distance between the two positions
        Vector3 direction = (newPosition - currentPosition).normalized;
        float distance = Vector3.Distance(currentPosition, newPosition);
        currentPosition.y = 0.5f;
        newPosition.y = 0.5f;
        Debug.Log($"Checking wall between: ({currentPosition.x},{currentPosition.y},{currentPosition.z}) and ({newPosition.x},{newPosition.y},{newPosition.z})");
        // Debug.Log("direction: " + direction.x + ", " + direction.y + ", " + direction.z);
        // Debug.Log("distance: " + distance);
        Debug.DrawRay(currentPosition, direction * distance, Color.red, 1.0f);
        // Perform a raycast to detect collisions
        if (Physics.Raycast(currentPosition, direction, out RaycastHit hit, distance))
        {
            // Check if the hit object is tagged as a wall
            if (hit.collider.CompareTag("wall"))
            {
                Debug.Log("Wall detected!");
                return true;
            }
        }

        return false;
    }
    /*
    private bool CheckForWallCollision(Vector3 currentPosition)
    {
        // Define a small radius for checking collision
        float checkRadius = 0.1f; // Adjust radius as needed
        currentPosition.y = 0f;
        // Get all colliders within the radius at the current position
        Collider[] colliders = Physics.OverlapSphere(currentPosition, checkRadius);

        // Iterate through the colliders to check for a "wall" tag
        foreach (Collider collider in colliders)
        {
            if (collider.CompareTag("wall") && collider.enabled)
            {
                Debug.Log("Collision with wall detected!");
                return true;
            }
        }

        // No collision detected
        Debug.Log($"No collision with wall at {currentPosition.x},{currentPosition.y},{currentPosition.z}.");
        return false;
    }
    */



    private void SendDataToPython(string data)
    {
        byte[] message = Encoding.ASCII.GetBytes(data + "\n");
        stream.Write(message, 0, message.Length);
        Debug.Log("Sending message: " + data);
    }

    private string ReceiveDataFromPython()
    {
        byte[] data = new byte[256];
        int bytes = stream.Read(data, 0, data.Length);
        return Encoding.ASCII.GetString(data, 0, bytes).Trim();
    }

    private void ResetPositions()
    {
        transform.position = initialAgentPosition;
        target.position = initialTargetPosition;
        Debug.Log("Environment reset to initial positions.");
        episodeDone = false;
    }

    private void CheckGoal()
    {
        float distanceToTarget = Vector3.Distance(transform.localPosition, target.localPosition);
        if (distanceToTarget < 0.1f)
        {
            episodeDone = true;
            Debug.Log("Agent reached the target.");
        }
    }
}
