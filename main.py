import mujoco
import mujoco.viewer


def main():
    with open("example.xml") as f:
        model = mujoco.MjModel.from_xml_string(f.read())
    mujoco.viewer.launch(model)
    print(model)


if __name__ == "__main__":
    main()
