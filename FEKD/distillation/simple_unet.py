Input: teacher: BiRefNet with larger model size; student: Smaller student model
Output: Knowledge Distillation (KD) Process for model training

1 size_teacher: Size of the teacher network
2 size_student: Size of the student network
3 N: Number of stages or layers for teaching assistant generation
4 M.acc: Accuracy of the teacher model
5 K: A hyperparameter controlling the stop epoch for KD
6 method = "KD"  // KD indicates Knowledge Distillation
7 EPOCHS = 100  // Number of training epochs
8 cmp_student = new student model  // Temporary model to compare
9 k = 0  // Counter for iteration epochs
10 for {i = 1, i ≤ N}  // Loop over N stages or layers in teacher network
11    teacher_features = get_teacher_features(teacher, i)  // Extract teacher features at stage i
12    student_features = get_student_features(student, i)  // Extract student features at stage i
13    loss = calculate_distillation_loss(teacher_features, student_features)  // Compute KD loss
14    optimize(student, loss)  // Optimize student network using KD loss
15    if {student.accuracy ≤ teacher.accuracy}  // Check if student's performance is worse
16        k = k + 1  // Increment failure counter if student is worse than teacher
17        if {k ≥ K}  // Check if we have reached the stopping condition for KD
18            method = "non_KD"  // Switch to non-KD training if KD has failed for K epochs
19        end if
20    else  // If student's performance is improving
21        k = 0  // Reset failure counter if student improves
22    end if
23 end for
24 if {method == "non_KD"}  // If method is switched to non-KD
25    train(student, non_KD)  // Train student model without KD
26 end if

Output: Trained student model
