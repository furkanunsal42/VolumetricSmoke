using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SmokeSourceVolumetric : MonoBehaviour
{
    public bool active;
    public float smoke_amount;
    public Vector3 smoke_initial_velocity;
    void Start()
    {

    }

    void FixedUpdate()
    {
        foreach(NavierStokesSolver solver in NavierStokesSolver.all_instances)
        {
            solver.add_density_worldcoord(transform.position, smoke_amount);
            solver.add_velocity_worldcoord(transform.position, smoke_initial_velocity);
        }
    }
}
