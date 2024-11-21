using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class MazeAgentSocket : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    public Vector3 target;
    public Vector3 start;
    private Vector3 agentCenter;
    private TcpListener server;
    public float moveSpeed = 1f;

    private Vector3 initialAgentPosition;
    private Vector3 initialTargetPosition;
    private bool episodeDone = false;
    private int steps = 0;
    private Vector3 origin = new Vector3(0f, 0f, 0f);

    private void Start()
    {
        GameObject targetObject = GameObject.FindGameObjectWithTag("target");
        GameObject startObject = GameObject.FindGameObjectWithTag("start");
        GameObject agentObject = gameObject;
        episodeDone = false;
        Time.timeScale = 0.5f;

        if (targetObject != null)
        {
            BoxCollider boxCollider = targetObject.GetComponent<BoxCollider>();
            if (boxCollider != null)
            {
                // Get the world position of the center of the Box 
                target = boxCollider.bounds.center;
            }
            else
            {
                Debug.LogError("TargetObject does not have a BoxCollider component.");
            }
        }
        else
        {
            Debug.LogError("TargetObject is null.");
        }

        if (startObject != null)
        {
            BoxCollider boxCollider = startObject.GetComponent<BoxCollider>();
            if (boxCollider != null)
            {
                // Get the world position of the center of the Box 
                start = boxCollider.bounds.center;
            }
            else
            {
                Debug.LogError("startObject does not have a BoxCollider component.");
            }
        }
        else
        {
            Debug.LogError("startObject is null.");
        }

        if (agentObject != null)
        {
            BoxCollider boxCollider = agentObject.GetComponent<BoxCollider>();
            if (boxCollider != null)
            {
                // Get the world position of the center of the Box 
                agentCenter = boxCollider.bounds.center;
            }
            else
            {
                Debug.LogError("agentObject does not have a BoxCollider component.");
            }
        }
        else
        {
            Debug.LogError("agentObject is null.");
        }

        initialAgentPosition = start;
        initialTargetPosition = target;
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
            string observations = CollectObservations();
            SendDataToPython(observations);

            string action = ReceiveDataFromPython();
            if (!string.IsNullOrEmpty(action))
            {
                Debug.Log("Received action from Python: " + action);
                if (action == "RESET")
                {
                    ResetPositions();
                }
                else
                {
                    PerformAction(action);
                }
            }

            steps++;
            if (steps >= 200 || episodeDone)
            {
                ResetPositions();
                steps = 0;
                episodeDone = false;
            }
        }
    }

    private string CollectObservations()
    {
        Vector3 agentRelativePosition = agentCenter - origin;
        Vector3 targetRelativePosition = target - origin;

        float agentX = agentRelativePosition.x;
        float agentY = agentRelativePosition.z;
        float targetX = targetRelativePosition.x;
        float targetY = targetRelativePosition.z;
        float done = episodeDone ? 1 : 0;
        Debug.Log($"collect obs: {agentX},{agentY},{targetX},{targetY}");
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
                move = new Vector3(-moveStep, 0, 0); // Move up (-ve x)
                targetRotation = Quaternion.Euler(0, 180, 0);
                break;
            case "2":
                move = new Vector3(0, 0, moveStep); // Move right (+ve z)
                targetRotation = Quaternion.Euler(0, 0, 0);
                break;
            case "3":
                move = new Vector3(moveStep, 0, 0); // Move down (+ve x)
                targetRotation = Quaternion.Euler(0, 270, 0);
                break;
            case "4":
                move = new Vector3(0, 0, -moveStep); // Move left (-ve z)
                targetRotation = Quaternion.Euler(0, 90, 0);
                break;
        }

        // Apply rotation first to face the direction
        transform.localRotation = targetRotation;

        // Apply movement
        if (move != Vector3.zero)
        {
            Vector3 newPosition = transform.position + move;
            Debug.Log("Trying to move to: " + newPosition.x + ", " + newPosition.z);

            if (CheckForWallCollision(agentCenter, agentCenter + move))
            {
                episodeDone = true;
                Debug.Log("Agent hit a wall. Ending episode.");
            }
            else
            {
                agentCenter = agentCenter + move;
                transform.position = newPosition;
                CheckGoal(agentCenter);
            }
        }
    }


    private bool CheckForWallCollision(Vector3 currentPosition, Vector3 newPosition)
    {
        // Calculate the direction and distance between the two positions
        Vector3 direction = newPosition - currentPosition;
        float distance = Vector3.Distance(currentPosition, newPosition);
        //currentPosition.y = 0.5f;
        Debug.Log("direction: " + direction.x + ", " + direction.y + ", " + direction.z);
        Debug.Log("distance: " + distance);
        Debug.DrawRay(currentPosition, direction * distance, Color.red, 5.0f);

        // Perform a raycast to detect collisions
        if (Physics.Raycast(currentPosition, direction, out RaycastHit hit, distance))
        {
            Debug.Log("Raycast direction: " + direction);
            Debug.Log("Raycast hit: " + (hit.collider != null ? hit.collider.name : "none"));

            // Check if the hit object is tagged as a wall
            if (hit.collider.CompareTag("wall"))
            {
                Debug.Log($"Wall detected between the positions: (${currentPosition.x},${currentPosition.y},${currentPosition.z}) and (${newPosition.x},${newPosition.y},${newPosition.z})");
                return true;
            }
        }

        return false;
    }

    private void SendDataToPython(string data)
    {
        byte[] message = Encoding.ASCII.GetBytes(data + "\n");
        stream.Write(message, 0, message.Length);
        Debug.Log("Sending msg: " + data);
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
        agentCenter = transform.position;
        agentCenter.y = 0;
        target = initialTargetPosition;
        Debug.Log("Environment reset to initial positions.");
        episodeDone = false;
    }

    private void CheckGoal(Vector3 newPosition)
    {
        float distanceToTarget = Vector3.Distance(transform.position, target);
        if (distanceToTarget < 0.1f)
        {
            episodeDone = true;
            Debug.Log("Agent reached the target.");
        }
    }
}
