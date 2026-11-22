# Limitations, in plain words

- **Side view, rigid objects, still camera.** Foresight models balls and boxes moving in the image plane. A ball
  rolling toward the camera, a cloth or a liquid is out of scope.
- **Depth is read from where things touch the table.** Objects that touch it on the same image line are at the
  same depth and can collide; a box lower in the image is in front and hides the ball instead of stopping it.
  A collision that glances off in depth (two balls deflecting sideways, a ball bouncing off a duck toward the
  back of the table) cannot be reproduced in 2D: the simulated balls bounce straight back.
- **Short motion.** In Physics-IQ the object often moves for only the last 0.2 to 0.6 s of the 3 s it is shown,
  so the fit leans on gravity and the object's known size.
- **Scale comes from Gemini's size estimate.** A wrong size (a 6.7 cm tennis ball called 10 cm) scales every
  distance by the same factor; falls are cross-checked against gravity, rolls are not.
- **Some physics comes from Gemini's reading, not from the clip.** Masses (which object gets knocked, and how
  far) and whether a surface is soft come from Hydrogen. Rolling resistance is a fixed 0.04 g for every ball,
  not measured.
- **A subset of Physics-IQ.** 15 of 66 scenes from the center camera in the main eval, 20 scenes from all three
  cameras in the wider eval, so the scores are not comparable with the public leaderboard.
- **Developed on the eval clips.** The physics changes after the hackathon were measured on the same 15
  center-view clips the main eval reports. The wider eval's left and right cameras and five extra scenes were
  not used while developing, so they are the fairer test.
- **The score likes stillness.** The Physics-IQ score compares where pixels change. Holding the last frame scores
  well, and so does an object that leaves the frame early, because a still picture matches a real scene that has
  come to rest, even when the object went the wrong way. The README reports endpoint error next to it for that
  reason.
- **Omni is slow and paid.** It took a median 38.9 s a clip in the eval and is billed per second of video, so
  the stage asks before calling it. Genie 4 streams its 5 s in about 15 to 30 s.
