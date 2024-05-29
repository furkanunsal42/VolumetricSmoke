using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmokeProof : MonoBehaviour
{
    [HideInInspector]
    public bool moved;
    public Transform old_transform;

    void Start()
    {
        
    }

    void Update()
    {
        
    }

    public Transform get_transform()
    {
        moved = false;

        return old_transform;
    }

    private void FixedUpdate()
    {
        if (transform != old_transform)
        {
            moved = true;
        }
    }
}
