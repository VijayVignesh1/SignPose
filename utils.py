import bpy
import cv2
import os
import numpy as np
def initialize_avatar():
    # Import the base avatar
    for o in bpy.context.scene.objects:
        if o.name == 'Cube':
            o.select_set(True)
        else:
            o.select_set(False)
    # Call the operator only once
    bpy.ops.object.delete()
    light1 = bpy.data.lights.new(name='light',type='POINT')
    light_object1 = bpy.data.objects.new(name="light", object_data=light1)
    light_object1.location=(-0.210575, -6.56262, 7.41522)
    light1.energy=1000
    light2 = bpy.data.lights.new(name='light.001',type='POINT')
    light_object2 = bpy.data.objects.new(name="light.001", object_data=light2)
    light_object2.location=(-0.210575, -6.56262, 4.99445)
    light2.energy=1000
    light3 = bpy.data.lights.new(name='light.002',type='POINT')
    light_object3 = bpy.data.objects.new(name="light.002", object_data=light3)
    light_object3.location=(-0.210575, -6.56262, 1.87751)
    light3.energy=1000
    bpy.ops.import_scene.fbx( filepath = "3D_Model/MaleModelSankit.fbx")
    armature = bpy.context.scene.objects['metarig']
    obj = bpy.context.selected_objects[0]
    obj.animation_data_clear()
    scene=bpy.context.scene
    bpy.ops.object.camera_add(location=(-0.0360, -12.9774, 4.5951), 
                            rotation=(1.5340, -0.0015,-0.0098))

    bpy.context.collection.objects.link(light_object1)
    bpy.context.collection.objects.link(light_object2)
    bpy.context.collection.objects.link(light_object3)
    bpy.context.view_layer.objects.active = light_object1
    bpy.context.view_layer.objects.active = light_object2
    bpy.context.view_layer.objects.active = light_object3
    return armature, scene

def two_d_pose_image(y, filename, armature, scene):
    """
    Merge ans save two 2d images for comparison.
    (Regressed 2d points and ground truth 2d points)
    Input is quaternions.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # initialize the camera parameters
    R=np.array([[1.0000, -0.0098,  0.0015],[0.0011, -0.0368, -0.9993],[0.0099,  0.9993, -0.0368]])
    T=np.array([[-0.0987], [4.1149], [13.1372]])
    f=50.0
    k=np.zeros((3,1))
    p=np.zeros((2,1))
    c=np.array([[125.0],[125.0]])

    # Rotate the avatar using the regressed quaternions
    final_output=[]
    obj = bpy.context.selected_objects[0]
    obj.animation_data_clear()
    poses=y   
    for bone in armature.pose.bones:
        if bone.name in poses:
            bone.rotation_quaternion.w=poses[bone.name][0]
            bone.rotation_quaternion.x=poses[bone.name][1]
            bone.rotation_quaternion.y=poses[bone.name][2]
            bone.rotation_quaternion.z=poses[bone.name][3]
            bone.keyframe_insert(data_path="rotation_quaternion" ,frame=0)
    bpy.context.scene.frame_set(0)

    bpy.context.scene.camera = bpy.data.objects['Camera.001']
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = ROOT_DIR + '/test.png'
    scene.render.resolution_x = 480 #perhaps set resolution in code
    scene.render.resolution_y = 640
    bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.ops.render.render(write_still=True)

    img=cv2.imread(ROOT_DIR + '/test.png')
    if not os.path.exists(ROOT_DIR + '/test_outputs'):
        os.makedirs(ROOT_DIR + '/test_outputs')
    cv2.imwrite(ROOT_DIR + "/test_outputs/"+filename, img)
    os.remove(ROOT_DIR + '/test.png')