#ifndef SIMULATION_H
#define SIMULATION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <QMutex>
#include "simparameters.h"
#include <QGLWidget>

class RigidBodyTemplate;
class RigidBodyInstance;
class Cloth;

typedef Eigen::Triplet<double> Tr;

class SimParameters;

class Simulation
{
public:
    Simulation(const SimParameters &params);
    ~Simulation();

    void takeSimulationStep();
    void initializeGL();

    void renderFloor();
    void renderObjects();
    void clearScene();
    void accelerateBody(double vx, double vy, double vz, double wx, double wy, double wz);

    Eigen::VectorXd computeForces();
    Eigen::VectorXd computeGravity();
    Eigen::VectorXd computeStretchingForce();
    Eigen::VectorXd computeDampingForce();
    Eigen::VectorXd computeContactForce();
    Eigen::VectorXd computeBendingForce();

private:
    void loadFloorTexture();
    void loadRigidBodies();

    const SimParameters &params_;
    QMutex renderLock_;

    double time_;
    GLuint floorTex_;

    RigidBodyTemplate * bodyTemplate_;
    RigidBodyInstance * bodyInstance_;
    int debug;

    Cloth * cloth_;
};

#endif // SIMULATION_H

