using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class MazeAgentSocket : MonoBehaviour
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
    private int steps = 0;
    private Vector3 origin = new Vector3(4.5144f, 0f, -10.581f);

    private void Start()
    {
        GameObject targetObject = GameObject.FindGameObjectWithTag("target");
        GameObject startObject = GameObject.FindGameObjectWithTag("start");
        episodeDone = false;
        Time.timeScale = 0.5f;

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
        initialTargetPosition = target.position;
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
        Vector3 agentRelativePosition = transform.localPosition - origin;
        Vector3 targetRelativePosition = target.localPosition - origin;

        float agentX = -agentRelativePosition.z;
        float agentY = -agentRelativePosition.x;
        float targetX = -targetRelativePosition.z;
        float targetY = -targetRelativePosition.x;
        float done = episodeDone ? 1 : 0;

        return $"{agentX},{agentY},{targetX},{targetY},{done}";
    }

    private void PerformAction(string action)
    {
        float moveStep = 1.0f; // Movement step size
        float rotationStep = 90.0f; // Rotation angle in degrees
        Vector3 move = Vector3.zero;
        Vector3 rotation = Vector3.zero;

        switch (action)
        {
            case "0": break; // No action
            case "1":
                move = new Vector3(0, 0, -moveStep); // Move forward (negative z-axis)
                rotation = new Vector3(0, -rotationStep, 0); // Rotate left (Y-axis)
                break;
            case "2":
                move = new Vector3(0, 0, moveStep); // Move backward (positive z-axis)
                rotation = new Vector3(0, rotationStep, 0); // Rotate right (Y-axis)
                break;
            case "3":
                move = new Vector3(-moveStep, 0, 0); // Move left (negative x-axis)
                rotation = new Vector3(-rotationStep, 0, 0); // Rotate down (X-axis)
                break;
            case "4":
                move = new Vector3(moveStep, 0, 0); // Move right (positive x-axis)
                rotation = new Vector3(rotationStep, 0, 0); // Rotate up (X-axis)
                break;
        }

        // Apply movement
        if (move != Vector3.zero)
        {
            Vector3 newPosition = transform.localPosition + move;
            Debug.Log("Trying to move to: " + newPosition.x + ", " + newPosition.z);

            if (CheckForWallCollision(newPosition))
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

        // Apply rotation
        if (rotation != Vector3.zero)
        {
            Quaternion newRotation = Quaternion.Euler(transform.localEulerAngles + rotation);
            Debug.Log("Rotating to: " + newRotation.eulerAngles);
            transform.localRotation = newRotation;
        }
    }


    private bool CheckForWallCollision(Vector3 position)
    {
        Collider[] colliders = Physics.OverlapBox(position, Vector3.one * 0.25f, Quaternion.identity);
        foreach (var collider in colliders)
        {
            if (collider.CompareTag("wall")) return true;
        }
        return false;
    }

    private void SendDataToPython(string data)
    {
        byte[] message = Encoding.ASCII.GetBytes(data + "\n");
        stream.Write(message, 0, message.Length);
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
        if (distanceToTarget < 0.75f)
        {
            episodeDone = true;
            Debug.Log("Agent reached the target.");
        }
    }
}
