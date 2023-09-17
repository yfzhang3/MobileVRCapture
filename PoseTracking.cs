using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoseTracking : MonoBehaviour
{
    // Start is called before the first frame update
    public UDPReceive udpReceive;
    public GameObject[] handPoints;
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        string data = udpReceive.data;

    
        print(data);
        string[] points = data.Split(',');
        print(points[0]); 

        //0        13      23
        //x1,y1,z1,x2,y2,z2,x3,y3,z3

        for (int i = 0; i<21; i++)
        {

            float x = float.Parse(points[i * 3])*-8;
            float y = float.Parse(points[i * 3 + 1])*-8;
            float z = float.Parse(points[i * 3 + 2])*-8;
            Vector3 whatever = new Vector3(x,y,z);
            handPoints[i].transform.localPosition = new Vector3(x, y, z);

        }
        
        

    }
}
