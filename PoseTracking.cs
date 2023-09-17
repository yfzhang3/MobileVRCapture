using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoseTracking : MonoBehaviour
{
    // Start is called before the first frame update
    public UDPReceive udpReceive;
    public GameObject[] bodyPoints;
    public GameObject[] leftHandPoints; 
    public GameObject[] rightHandPoints; 

    private float desiredDuration = 3f;
    private float elapsedTime;

    // Debug.Log("rightHandData.Count: " + rightHandData.Count);
    // Debug.Log("rightHandPoints.Length: " + rightHandPoints.Length);


    List<Vector3> bodyData = new List<Vector3>();
    List<Vector3> leftHandData = new List<Vector3>();
    List<Vector3> rightHandData = new List<Vector3>();
    void Start()
    {
        for (int y = 0; y < 21; y++) {
            Vector3 location = bodyPoints[y].transform.localPosition;
            bodyData.Add(location);
        }

        for (int y = 0; y < 10; y++)
        {
            Vector3 location = leftHandPoints[y].transform.localPosition;
            leftHandData.Add(location);
        }

        for (int y = 0; y < 10; y++)
        {
            Vector3 location = rightHandPoints[y].transform.localPosition;
            rightHandData.Add(location);
        }
    }

    // Update is called once per frame
    void Update()
    {
        // try: elapsedTime += Time.deltaTime; for smoother transition
        elapsedTime += 5f; 

        string data = udpReceive.data;
        Debug.Log("Received data: " + data);
        string[] points = data.Split(',');
        Debug.Log("Number of points: " + points.Length);

        // Check if the points array has the expected length
        if (points.Length != 63) // Adjust the length as needed
        {
            Debug.LogError("Received data has an unexpected format. Expected 63 elements, but received " + points.Length + " elements.");
            return; // Exit the method to prevent further processing
        }

        // Update body points
        for (int i = 0; i < bodyPoints.Length; i++)
        {
            // Attempt to parse the values, and handle any parsing errors
            float x, y, z;
            if (!float.TryParse(points[i * 3], out x) ||
                !float.TryParse(points[i * 3 + 1], out y) ||
                !float.TryParse(points[i * 3 + 2], out z))
            {

            bodyPoints[i].transform.localPosition = Vector3.Lerp(bodyData[i], endingPosition, elapsedTime/desiredDuration);
            
            bodyData[i] = endingPosition;
        }  
        print("finish body");

        // Update left hand points
        int startIndex = bodyPoints.Length; 
        
        for (int i = 0; i < leftHandPoints.Length; i++)
        {
            float x = float.Parse(points[startIndex + i * 3]) * -5;
            float y = float.Parse(points[startIndex + i * 3 + 1]) * -5;
            float z = float.Parse(points[startIndex + i * 3 + 2]) * -5;

            Vector3 endingPosition = new Vector3(x, y, z);

            leftHandPoints[i].transform.localPosition = Vector3.Lerp(leftHandData[i], endingPosition, elapsedTime/desiredDuration);

            leftHandData[i] = endingPosition;
        }
        print("finish left hand");
    }

        // Update right hand points
        startIndex = leftHandPoints.Length + bodyPoints.Length * 3; 
        for (int i = 0; i < rightHandPoints.Length; i++)
        {
            float x = float.Parse(points[startIndex + i * 3]) * -5;
            float y = float.Parse(points[startIndex + i * 3 + 1]) * -5;
            float z = float.Parse(points[startIndex + i * 3 + 2]) * -5;

            Vector3 endingPosition = new Vector3(x, y, z);

            rightHandPoints[i].transform.localPosition = Vector3.Lerp(rightHandData[i], endingPosition, elapsedTime / desiredDuration);

            rightHandData[i] = endingPosition;
        }
        print("finish right hand");

}

