import rospy
import moveit_commander
import sys
import time
from geometry_msgs.msg import PoseStamped

def load_mesh(mesh_file, mesh_name="pointcloud_mesh", frame_id="world"):
    rospy.init_node("scene_updater", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    scene = moveit_commander.PlanningSceneInterface()
    #rospy.sleep(2)

    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.orientation.w = 1.0

    scene.remove_world_object(mesh_name)
    #rospy.sleep(1)
    scene.add_mesh(mesh_name, pose, mesh_file)
    #rospy.sleep(1)

    #print(f"Added {mesh_file} as {mesh_name} to planning scene.")

if __name__ == "__main__":
    load_mesh("mesh.stl")
